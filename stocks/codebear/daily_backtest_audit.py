#!/usr/bin/env python3
"""
日频权益曲线审计 — 月频调仓 + 日频净值追踪
代码熊 🐻

目的：修复月频回测中 MaxDD 严重低估的缺陷
方法：复用 v9f/v9g 的选股逻辑，用日频 close-to-close 追踪净值

严格约束：
- 今日净值 = 昨日收盘→今日收盘的涨跌 (close[i-1]→close[i])
- 月末调仓信号基于当月末收盘价，下个交易日才买 (T+1 滑点)
- 时间段 2015-01 ~ 2025-12
"""

import json, warnings, sys
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# ─── Import v9f and v9g logic ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

# We'll import the modules dynamically
import importlib.util

def load_strategy_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ─── Data Loading ──────────────────────────────────────────────────────────

def load_csv(fp):
    df = pd.read_csv(fp)
    c = 'Date' if 'Date' in df.columns else df.columns[0]
    df[c] = pd.to_datetime(df[c])
    df = df.set_index(c).sort_index()
    if 'Close' in df.columns:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    return df


def load_stocks(tickers):
    d = {}
    for t in tickers:
        f = STOCK_CACHE / f"{t}.csv"
        if not f.exists() or f.stat().st_size < 500:
            continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200:
                d[t] = df['Close'].dropna()
        except:
            pass
    return pd.DataFrame(d)


# ─── Daily Backtest Engine ─────────────────────────────────────────────────

def run_daily_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                       select_fn, apply_overlays_fn, get_spy_vol_fn,
                       start='2015-01-01', end='2025-12-31', 
                       cost=0.0015, stop_loss=None):
    """
    月频调仓 + 日频净值追踪 + 可选月中止损
    
    核心改进：用日频 close-to-close 追踪净值，揭示月中隐藏的回撤
    
    stop_loss: 单月跌幅阈值 (e.g. -0.15)，超过则切换到 SHY
               设为 None 不启用止损
    """
    # 构建日频价格矩阵（包含所有资产）
    all_daily = close_df.loc[start:end].dropna(how='all')
    
    # 获取每月末调仓日期
    month_ends = all_daily.resample('ME').last().index
    # 过滤掉不在交易日的月末
    trading_days = all_daily.index
    
    val = 1.0
    peak = 1.0
    equity_vals = []
    equity_dates = []
    
    current_weights = {}
    prev_hold = set()
    prev_w = {}
    month_start_val = 1.0
    dd_for_signal = 0.0  # DD at signal time (month-end)
    
    processed_months = set()
    stop_loss_active = False
    
    # 记录月频权益用于对比
    monthly_vals = []
    monthly_dates = []
    
    for day_idx, day in enumerate(trading_days):
        # ─── 月末调仓检查 ─────────────────────────────
        # 实际上我们在月末后的第一个交易日执行调仓（T+1滑点）
        # 找到上一个月末
        past_month_ends = month_ends[month_ends < day]
        
        if len(past_month_ends) > 0:
            last_me = past_month_ends[-1]
            
            # 下一个交易日 = month_end 之后的第一个交易日
            next_days_after_me = trading_days[trading_days > last_me]
            if len(next_days_after_me) > 0:
                execution_day = next_days_after_me[0]
            else:
                execution_day = None
            
            if execution_day is not None and day == execution_day and last_me not in processed_months:
                # 用月末数据生成信号，今天执行
                dd_for_signal = (val - peak) / peak if peak > 0 else 0
                spy_vol = get_spy_vol_fn(sig, last_me)
                
                new_weights = select_fn(sig, sectors, last_me, prev_hold, gld_p, gdx_p)
                new_weights = apply_overlays_fn(new_weights, spy_vol, dd_for_signal)
                
                # 计算换手成本
                all_t = set(new_weights) | set(prev_w)
                turnover = sum(abs(new_weights.get(t,0) - prev_w.get(t,0)) for t in all_t) / 2
                val *= (1 - turnover * cost * 2)
                
                current_weights = new_weights.copy()
                prev_w = new_weights.copy()
                prev_hold = {k for k in new_weights if k not in ('GLD','GDX','GDXJ','SHY')}
                
                month_start_val = val
                stop_loss_active = False
                processed_months.add(last_me)
                
                # 记录月频数据点
                monthly_vals.append(val)
                monthly_dates.append(day)
        
        # ─── 每日净值更新 ───────────────────────────────
        if day_idx == 0:
            equity_vals.append(val)
            equity_dates.append(day)
            continue
        
        prev_day = trading_days[day_idx - 1]
        day_ret = 0.0
        invested = 0.0
        
        for ticker, w in current_weights.items():
            # 获取对应的价格序列
            if ticker == 'GLD':
                series = gld_p
            elif ticker == 'GDX':
                series = gdx_p
            elif ticker == 'GDXJ':
                series = gdxj_p
            elif ticker == 'SHY':
                series = shy_p
            elif ticker in close_df.columns:
                series = close_df[ticker]
            else:
                continue
            
            # close[i-1] → close[i] 的日收益
            if prev_day in series.index and day in series.index:
                p_prev = series.loc[prev_day]
                p_today = series.loc[day]
                if pd.notna(p_prev) and pd.notna(p_today) and p_prev > 0:
                    ticker_ret = p_today / p_prev - 1
                    day_ret += ticker_ret * w
                    invested += w
            
        # 未投资部分假设持有 SHY
        cash_frac = max(1.0 - invested, 0.0)
        if cash_frac > 0.01 and shy_p is not None:
            if prev_day in shy_p.index and day in shy_p.index:
                sp = shy_p.loc[prev_day]
                st = shy_p.loc[day]
                if pd.notna(sp) and pd.notna(st) and sp > 0:
                    day_ret += (st / sp - 1) * cash_frac
        
        val *= (1 + day_ret)
        peak = max(peak, val)
        
        # ─── 月中止损检查 ───────────────────────────────
        if stop_loss is not None and not stop_loss_active and month_start_val > 0:
            month_ret = val / month_start_val - 1
            if month_ret < stop_loss:
                # 切换到 SHY 防御，本月剩余时间
                current_weights = {'SHY': 1.0}
                stop_loss_active = True
        
        equity_vals.append(val)
        equity_dates.append(day)
    
    equity = pd.Series(equity_vals, index=pd.DatetimeIndex(equity_dates))
    monthly_eq = pd.Series(monthly_vals, index=pd.DatetimeIndex(monthly_dates)) if monthly_dates else pd.Series(dtype=float)
    
    return equity, monthly_eq


def compute_metrics_daily(eq, rf=0.04):
    """从日频权益曲线计算指标"""
    if len(eq) < 30:
        return dict(cagr=0, max_dd=0, sharpe=0, calmar=0, composite=0)
    
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs < 0.5:
        return dict(cagr=0, max_dd=0, sharpe=0, calmar=0, composite=0)
    
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1
    
    # 日频 MaxDD
    running_max = eq.cummax()
    drawdown = (eq - running_max) / running_max
    max_dd = drawdown.min()
    
    # 日频 Sharpe (年化)
    daily_rets = eq.pct_change().dropna()
    ann_ret = daily_rets.mean() * 252
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    composite = sharpe * 0.4 + calmar * 0.4 + cagr * 0.2
    
    # 找到最大回撤的时间区间
    dd_end_idx = drawdown.idxmin()
    dd_start_idx = eq.loc[:dd_end_idx].idxmax()
    
    return dict(
        cagr=float(cagr), 
        max_dd=float(max_dd), 
        sharpe=float(sharpe), 
        calmar=float(calmar),
        composite=float(composite),
        max_dd_start=str(dd_start_idx.date()),
        max_dd_end=str(dd_end_idx.date()),
        ann_vol=float(ann_vol),
        final_val=float(eq.iloc[-1])
    )


def compute_metrics_monthly(eq):
    """从月频权益曲线计算指标（与原始回测一致的方法）"""
    if len(eq) < 3:
        return dict(cagr=0, max_dd=0, sharpe=0, calmar=0, composite=0)
    
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs < 0.5:
        return dict(cagr=0, max_dd=0, sharpe=0, calmar=0, composite=0)
    
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1
    mo = eq.pct_change().dropna()
    sharpe = mo.mean() / mo.std() * np.sqrt(12) if mo.std() > 0 else 0
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    calmar = cagr / abs(dd) if dd != 0 else 0
    composite = sharpe * 0.4 + calmar * 0.4 + cagr * 0.2
    
    return dict(cagr=float(cagr), max_dd=float(dd), sharpe=float(sharpe), 
                calmar=float(calmar), composite=float(composite))


# ─── Run Original Monthly Backtest (for comparison) ───────────────────────

def run_monthly_backtest_orig(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                              select_fn, apply_overlays_fn, get_spy_vol_fn,
                              start='2015-01-01', end='2025-12-31', cost=0.0015,
                              use_shy=True):
    """复制原始月频回测逻辑（与 v9f/v9g 的 run_backtest 一致）"""
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        spy_vol = get_spy_vol_fn(sig, dt)
        w = select_fn(sig, sectors, dt, prev_h, gld_p, gdx_p)
        w = apply_overlays_fn(w, spy_vol, dd)

        all_t = set(w) | set(prev_w)
        to = sum(abs(w.get(t,0)-prev_w.get(t,0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in ('GLD','GDX','GDXJ')}

        invested = sum(w.values()); cash_frac = max(1.0 - invested, 0.0)
        ret = 0.0
        for t, wt in w.items():
            if   t == 'GLD':  s = gld_p.loc[dt:ndt].dropna()
            elif t == 'GDX':  s = gdx_p.loc[dt:ndt].dropna()
            elif t == 'GDXJ': s = gdxj_p.loc[dt:ndt].dropna()
            elif t in close_df.columns: s = close_df[t].loc[dt:ndt].dropna()
            else: continue
            if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * wt
        if use_shy and cash_frac > 0 and shy_p is not None:
            s = shy_p.loc[dt:ndt].dropna()
            if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * cash_frac
        ret -= to * cost * 2
        val *= (1 + ret)
        if val > peak: peak = val
        vals.append(val); dates.append(ndt)

    eq = pd.Series(vals, index=pd.DatetimeIndex(dates))
    return eq


# ─── Main Audit ────────────────────────────────────────────────────────────

def audit_stock_strategy(version_name, strategy_module, close_df, sig, sectors, 
                          gld_p, gdx_p, gdxj_p, shy_p):
    """审计单个策略：月频 vs 日频 对比"""
    print(f"\n{'='*70}")
    print(f"🔍 审计 {version_name}")
    print(f"{'='*70}")
    
    select_fn = strategy_module.select
    apply_overlays_fn = strategy_module.apply_overlays
    get_spy_vol_fn = strategy_module.get_spy_vol
    
    # 1. 原始月频回测
    print("  [1/4] 原始月频回测...")
    eq_monthly = run_monthly_backtest_orig(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
        select_fn, apply_overlays_fn, get_spy_vol_fn
    )
    m_monthly = compute_metrics_monthly(eq_monthly)
    
    # 2. 日频回测 - 无止损
    print("  [2/4] 日频回测 (无止损)...")
    eq_daily_nostp, _ = run_daily_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
        select_fn, apply_overlays_fn, get_spy_vol_fn,
        stop_loss=None
    )
    m_daily_nostp = compute_metrics_daily(eq_daily_nostp)
    
    # 3. 日频回测 - 止损 -15%
    print("  [3/4] 日频回测 (止损 -15%)...")
    eq_daily_sl15, _ = run_daily_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
        select_fn, apply_overlays_fn, get_spy_vol_fn,
        stop_loss=-0.15
    )
    m_daily_sl15 = compute_metrics_daily(eq_daily_sl15)
    
    # 4. 日频回测 - 止损 -10%
    print("  [4/4] 日频回测 (止损 -10%)...")
    eq_daily_sl10, _ = run_daily_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
        select_fn, apply_overlays_fn, get_spy_vol_fn,
        stop_loss=-0.10
    )
    m_daily_sl10 = compute_metrics_daily(eq_daily_sl10)
    
    # 输出结果
    print(f"\n  {'指标':<20} {'月频':>12} {'日频(无止损)':>14} {'日频(-15%)':>14} {'日频(-10%)':>14}")
    print(f"  {'-'*74}")
    print(f"  {'CAGR':<20} {m_monthly['cagr']:>11.2%} {m_daily_nostp['cagr']:>13.2%} {m_daily_sl15['cagr']:>13.2%} {m_daily_sl10['cagr']:>13.2%}")
    print(f"  {'MaxDD':<20} {m_monthly['max_dd']:>11.2%} {m_daily_nostp['max_dd']:>13.2%} {m_daily_sl15['max_dd']:>13.2%} {m_daily_sl10['max_dd']:>13.2%}")
    print(f"  {'Sharpe':<20} {m_monthly['sharpe']:>11.2f} {m_daily_nostp['sharpe']:>13.2f} {m_daily_sl15['sharpe']:>13.2f} {m_daily_sl10['sharpe']:>13.2f}")
    print(f"  {'Calmar':<20} {m_monthly['calmar']:>11.2f} {m_daily_nostp['calmar']:>13.2f} {m_daily_sl15['calmar']:>13.2f} {m_daily_sl10['calmar']:>13.2f}")
    print(f"  {'Composite':<20} {m_monthly['composite']:>11.4f} {m_daily_nostp['composite']:>13.4f} {m_daily_sl15['composite']:>13.4f} {m_daily_sl10['composite']:>13.4f}")
    
    if 'max_dd_start' in m_daily_nostp:
        print(f"\n  日频最大回撤区间(无止损): {m_daily_nostp['max_dd_start']} → {m_daily_nostp['max_dd_end']}")
    
    dd_gap = m_daily_nostp['max_dd'] - m_monthly['max_dd']
    print(f"\n  ⚠️  MaxDD 差距: 月频 {m_monthly['max_dd']:.2%} vs 日频 {m_daily_nostp['max_dd']:.2%} (差 {dd_gap:.2%})")
    
    return {
        'monthly': m_monthly,
        'daily_no_stop': m_daily_nostp,
        'daily_sl15': m_daily_sl15,
        'daily_sl10': m_daily_sl10,
        'equity_daily': eq_daily_nostp,
        'equity_monthly': eq_monthly,
    }


# ─── BestCrypto Daily Audit ───────────────────────────────────────────────

def audit_bestcrypto():
    """审计 BestCrypto 策略"""
    print(f"\n{'='*70}")
    print(f"🔍 审计 BestCrypto (Crypto Dual Momentum)")
    print(f"{'='*70}")
    
    btc = pd.read_csv(CACHE / "BTC_USD.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    gld = pd.read_csv(CACHE / "GLD.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    
    # 尝试加载 ETH
    eth_path = CACHE / "ETH_USD.csv"
    if eth_path.exists():
        eth = pd.read_csv(eth_path, parse_dates=["Date"]).set_index("Date").sort_index()
    else:
        print("  ⚠️ ETH 数据不存在，仅测试 BTC v7f")
        eth = None
    
    # 共同日期
    common = btc.index.intersection(gld.index)
    if eth is not None:
        common = common.intersection(eth.index)
    common = common.sort_values()
    
    # 过滤 2015-2025
    common = common[(common >= '2015-01-01') & (common <= '2025-12-31')]
    
    btc_c = btc.loc[common, 'Close'].values.astype(float)
    gld_c = gld.loc[common, 'Close'].values.astype(float)
    dates = common.to_list()
    
    if eth is not None:
        eth_c = eth.loc[common, 'Close'].values.astype(float)
    
    print(f"  数据范围: {dates[0].date()} → {dates[-1].date()} ({len(dates)} 天)")
    
    # ─── BestCrypto 策略 (日频，已经是日频的) ──────
    # 这个策略本身就是日频的，但我们重新计算指标
    
    n = len(btc_c)
    
    # BestCrypto 策略
    if eth is not None:
        equity_best = np.zeros(n)
        equity_best[0] = 10000.0
        
        for i in range(1, n):
            btc_ret = btc_c[i] / btc_c[i-1] - 1
            eth_ret = eth_c[i] / eth_c[i-1] - 1
            gld_ret = gld_c[i] / gld_c[i-1] - 1
            
            lb = min(i, 90)
            btc_mom = btc_c[i-1] / btc_c[max(0, i-1-lb)] - 1
            eth_mom = eth_c[i-1] / eth_c[max(0, i-1-lb)] - 1
            gld_mom = gld_c[i-1] / gld_c[max(0, i-1-lb)] - 1
            
            moms = {"btc": btc_mom, "eth": eth_mom, "gld": gld_mom}
            best = max(moms, key=moms.get)
            
            if best == "btc" and btc_mom > 0:
                w_btc, w_eth, w_gld = 0.70, 0.15, 0.10
            elif best == "eth" and eth_mom > 0:
                w_btc, w_eth, w_gld = 0.15, 0.65, 0.10
            elif best == "gld":
                w_btc, w_eth, w_gld = 0.15, 0.10, 0.55
            else:
                w_btc, w_eth, w_gld = 0.15, 0.10, 0.35
            
            port_ret = w_btc * btc_ret + w_eth * eth_ret + w_gld * gld_ret
            equity_best[i] = equity_best[i-1] * (1 + port_ret)
        
        eq_best = pd.Series(equity_best, index=pd.DatetimeIndex(dates))
    
    # BTC v7f 策略
    equity_v7f = np.zeros(n)
    equity_v7f[0] = 10000.0
    sma200 = pd.Series(btc_c).rolling(200).mean().values
    
    halving_dates = [pd.Timestamp("2016-07-09"), pd.Timestamp("2020-05-11"), pd.Timestamp("2024-04-19")]
    
    def months_since_halving(d):
        for h in reversed(halving_dates):
            if d >= h:
                return (d - h).days / 30.44
        return 999
    
    for i in range(1, n):
        if np.isnan(sma200[i]):
            equity_v7f[i] = equity_v7f[i-1]
            continue
        
        btc_ret = btc_c[i] / btc_c[i-1] - 1
        gld_ret = gld_c[i] / gld_c[i-1] - 1
        
        lb3 = min(i, 63)
        lb6 = min(i, 126)
        btc_mom3 = btc_c[i-1] / btc_c[max(0, i-1-lb3)] - 1
        btc_mom6 = btc_c[i-1] / btc_c[max(0, i-1-lb6)] - 1
        gld_mom3 = gld_c[i-1] / gld_c[max(0, i-1-lb3)] - 1
        gld_mom6 = gld_c[i-1] / gld_c[max(0, i-1-lb6)] - 1
        
        btc_mom = 0.5 * btc_mom3 + 0.5 * btc_mom6
        gld_mom = 0.5 * gld_mom3 + 0.5 * gld_mom6
        
        mayer = btc_c[i] / sma200[i]
        msh = months_since_halving(dates[i])
        
        btc_pos = btc_mom > 0
        gld_pos = gld_mom > 0
        
        if btc_pos and gld_pos:
            if btc_mom > gld_mom:
                w_btc, w_gld = 0.80, 0.15
            else:
                w_btc, w_gld = 0.50, 0.40
        elif btc_pos and not gld_pos:
            w_btc, w_gld = 0.85, 0.05
        elif not btc_pos and gld_pos:
            w_btc, w_gld = 0.25, 0.50
        else:
            w_btc, w_gld = 0.20, 0.30
        
        if msh < 18:
            w_btc = max(w_btc, 0.50)
        
        if mayer > 3.5:
            w_btc = min(w_btc, 0.35)
        elif mayer > 2.4:
            w_btc = min(w_btc, 0.60)
        
        port_ret = w_btc * btc_ret + w_gld * gld_ret
        equity_v7f[i] = equity_v7f[i-1] * (1 + port_ret)
    
    eq_v7f = pd.Series(equity_v7f, index=pd.DatetimeIndex(dates))
    
    # ─── 月频采样 vs 日频对比 ──────────────────────
    # 模拟月频采样（只看月末）
    def monthly_sampled_metrics(eq):
        eq_monthly = eq.resample('ME').last().dropna()
        return compute_metrics_monthly(eq_monthly)
    
    def daily_metrics(eq):
        return compute_metrics_daily(eq)
    
    results = {}
    
    # BTC v7f
    m_v7f_monthly = monthly_sampled_metrics(eq_v7f)
    m_v7f_daily = daily_metrics(eq_v7f)
    
    print(f"\n  === BTC v7f DualMom ===")
    print(f"  {'指标':<15} {'月频采样':>12} {'日频真实':>12}")
    print(f"  {'-'*39}")
    print(f"  {'CAGR':<15} {m_v7f_monthly['cagr']:>11.2%} {m_v7f_daily['cagr']:>11.2%}")
    print(f"  {'MaxDD':<15} {m_v7f_monthly['max_dd']:>11.2%} {m_v7f_daily['max_dd']:>11.2%}")
    print(f"  {'Sharpe':<15} {m_v7f_monthly['sharpe']:>11.2f} {m_v7f_daily['sharpe']:>11.2f}")
    print(f"  {'Calmar':<15} {m_v7f_monthly['calmar']:>11.2f} {m_v7f_daily['calmar']:>11.2f}")
    print(f"  {'Composite':<15} {m_v7f_monthly['composite']:>11.4f} {m_v7f_daily['composite']:>11.4f}")
    
    if 'max_dd_start' in m_v7f_daily:
        print(f"  最大回撤区间: {m_v7f_daily['max_dd_start']} → {m_v7f_daily['max_dd_end']}")
    
    results['btc_v7f'] = {'monthly': m_v7f_monthly, 'daily': m_v7f_daily}
    
    # 重点关注特定时期
    for period_name, period_start, period_end in [
        ("2020年3月 (COVID)", "2020-02-15", "2020-04-15"),
        ("2022年全年 (Crypto Winter)", "2022-01-01", "2022-12-31"),
        ("2018年 (BTC Bear)", "2018-01-01", "2018-12-31"),
    ]:
        sub = eq_v7f.loc[period_start:period_end]
        if len(sub) > 10:
            sub_peak = sub.cummax()
            sub_dd = ((sub - sub_peak) / sub_peak).min()
            print(f"  {period_name}: MaxDD = {sub_dd:.2%}")
    
    if eth is not None:
        m_best_monthly = monthly_sampled_metrics(eq_best)
        m_best_daily = daily_metrics(eq_best)
        
        print(f"\n  === BestCrypto (BTC+ETH+GLD) ===")
        print(f"  {'指标':<15} {'月频采样':>12} {'日频真实':>12}")
        print(f"  {'-'*39}")
        print(f"  {'CAGR':<15} {m_best_monthly['cagr']:>11.2%} {m_best_daily['cagr']:>11.2%}")
        print(f"  {'MaxDD':<15} {m_best_monthly['max_dd']:>11.2%} {m_best_daily['max_dd']:>11.2%}")
        print(f"  {'Sharpe':<15} {m_best_monthly['sharpe']:>11.2f} {m_best_daily['sharpe']:>11.2f}")
        print(f"  {'Calmar':<15} {m_best_monthly['calmar']:>11.2f} {m_best_daily['calmar']:>11.2f}")
        print(f"  {'Composite':<15} {m_best_monthly['composite']:>11.4f} {m_best_daily['composite']:>11.4f}")
        
        if 'max_dd_start' in m_best_daily:
            print(f"  最大回撤区间: {m_best_daily['max_dd_start']} → {m_best_daily['max_dd_end']}")
        
        for period_name, period_start, period_end in [
            ("2020年3月 (COVID)", "2020-02-15", "2020-04-15"),
            ("2022年全年 (Crypto Winter)", "2022-01-01", "2022-12-31"),
        ]:
            sub = eq_best.loc[period_start:period_end]
            if len(sub) > 10:
                sub_peak = sub.cummax()
                sub_dd = ((sub - sub_peak) / sub_peak).min()
                print(f"  {period_name}: MaxDD = {sub_dd:.2%}")
        
        results['bestcrypto'] = {'monthly': m_best_monthly, 'daily': m_best_daily}
    
    return results


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("🐻 日频权益曲线审计 — 修复 MaxDD 低估缺陷")
    print("=" * 70)
    
    # 加载数据
    print("\n📂 加载数据...")
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    print(f"  加载 {len(close_df.columns)} 支股票")
    
    # 预计算信号
    print("📊 预计算信号...")
    
    # 加载 v9f
    v9f_path = Path(__file__).parent / "momentum_v9f_final.py"
    v9f_mod = load_strategy_module("v9f", v9f_path)
    sig_v9f = v9f_mod.precompute(close_df)
    
    # 加载 v9g
    v9g_path = Path(__file__).parent / "momentum_v9g_final.py"
    v9g_mod = load_strategy_module("v9g", v9g_path)
    sig_v9g = v9g_mod.precompute(close_df)
    
    all_results = {}
    
    # 审计 v9f
    r_v9f = audit_stock_strategy("v9f (GDXJ波动率+GDX精调)", v9f_mod,
                                  close_df, sig_v9f, sectors, gld_p, gdx_p, gdxj_p, shy_p)
    all_results['v9f'] = r_v9f
    
    # 审计 v9g
    r_v9g = audit_stock_strategy("v9g (动态行业集中度)", v9g_mod,
                                  close_df, sig_v9g, sectors, gld_p, gdx_p, gdxj_p, shy_p)
    all_results['v9g'] = r_v9g
    
    # 审计 BestCrypto
    r_crypto = audit_bestcrypto()
    all_results['crypto'] = r_crypto
    
    # ─── 总结 ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"📊 总结：月频 vs 日频 MaxDD 对比")
    print(f"{'='*70}")
    
    print(f"\n{'策略':<25} {'月频MaxDD':>12} {'日频MaxDD':>12} {'差距':>10} {'低估倍数':>10}")
    print(f"{'-'*69}")
    
    for name, key in [('Stock v9f', 'v9f'), ('Stock v9g', 'v9g')]:
        m_mo = all_results[key]['monthly']['max_dd']
        m_da = all_results[key]['daily_no_stop']['max_dd']
        gap = m_da - m_mo
        ratio = m_da / m_mo if m_mo != 0 else float('inf')
        print(f"  {name:<23} {m_mo:>11.2%} {m_da:>11.2%} {gap:>9.2%} {ratio:>9.2f}x")
    
    if 'btc_v7f' in all_results.get('crypto', {}):
        m_mo = all_results['crypto']['btc_v7f']['monthly']['max_dd']
        m_da = all_results['crypto']['btc_v7f']['daily']['max_dd']
        gap = m_da - m_mo
        ratio = m_da / m_mo if m_mo != 0 else float('inf')
        print(f"  {'BTC v7f':<23} {m_mo:>11.2%} {m_da:>11.2%} {gap:>9.2%} {ratio:>9.2f}x")
    
    if 'bestcrypto' in all_results.get('crypto', {}):
        m_mo = all_results['crypto']['bestcrypto']['monthly']['max_dd']
        m_da = all_results['crypto']['bestcrypto']['daily']['max_dd']
        gap = m_da - m_mo
        ratio = m_da / m_mo if m_mo != 0 else float('inf')
        print(f"  {'BestCrypto':<23} {m_mo:>11.2%} {m_da:>11.2%} {gap:>9.2%} {ratio:>9.2f}x")
    
    # 止损效果
    print(f"\n{'策略':<25} {'日频无止损':>12} {'日频-15%止损':>14} {'日频-10%止损':>14}")
    print(f"{'-'*65}")
    for name, key in [('Stock v9f', 'v9f'), ('Stock v9g', 'v9g')]:
        m0 = all_results[key]['daily_no_stop']['max_dd']
        m15 = all_results[key]['daily_sl15']['max_dd']
        m10 = all_results[key]['daily_sl10']['max_dd']
        print(f"  {name:<23} {m0:>11.2%} {m15:>13.2%} {m10:>13.2%}")
    
    # 保存详细结果
    output = {}
    for key in ['v9f', 'v9g']:
        output[key] = {
            'monthly': all_results[key]['monthly'],
            'daily_no_stop': {k: v for k, v in all_results[key]['daily_no_stop'].items() if k != 'equity'},
            'daily_sl15': {k: v for k, v in all_results[key]['daily_sl15'].items() if k != 'equity'},
            'daily_sl10': {k: v for k, v in all_results[key]['daily_sl10'].items() if k != 'equity'},
        }
    if 'crypto' in all_results:
        output['crypto'] = all_results['crypto']
    
    result_file = Path(__file__).parent / "daily_backtest_audit_results.json"
    result_file.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n💾 详细结果 → {result_file}")
    
    return all_results


if __name__ == '__main__':
    results = main()
