#!/usr/bin/env python3
"""
Hybrid v4 Daily — Stock v11b + Crypto v3 + Direction C(短动量)
代码熊 🐻 | 2026-02-22

核心创新（相比 Hybrid v3）：
  1. 股票组件升级：v9j → v11b
     - v11b: CAGR=32.7%, Sharpe=1.96, MaxDD=-20.78%（审计通过）
     - 比 v9j 高: Sharpe 1.39→1.96, 日频 MaxDD -25.1%→-20.78%
     - IS Sharpe 更高 → WF=OOS/IS Sharpe 更有余量 → 可以用更多 crypto

  2. Crypto 组件：保留 v3 的三重保护（200dMA + DD + vol targeting）
     - 另增 Direction C：测试 45d 短动量窗口（vs 原 90d）
     - 较短窗口 → 2022 年反转时更快退出

  3. WF 固定日期边界：
     - IS=2015-01-01 ~ 2020-12-31
     - OOS=2021-01-01 ~ 2025-12-31

  4. 新 Composite = Sharpe×0.25 + Calmar×0.25 + min(CAGR,2.0)×0.50

  5. 突破条件：CAGR>40% + WF>0.60 + 日频MaxDD<30% → 标注【重大突破】
"""

import json
import sys
import importlib.util
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE        = Path(__file__).resolve().parents[2]
CACHE       = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"
V11B_PATH   = BASE / "stocks" / "codebear" / "momentum_v11b_final.py"

# 动态导入 v11b 模块
spec = importlib.util.spec_from_file_location("v11b_mod", V11B_PATH)
v11b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v11b)

# Crypto v3 过滤参数
BTC_MA_WINDOW  = 200
BTC_DD_EXIT    = -0.25
BTC_DD_REENTRY = -0.15
BTC_VOL_WINDOW = 60
BTC_VOL_TARGET = 0.80
BTC_VOL_MIN    = 0.40
BTC_VOL_MAX    = 1.50


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════════════
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
        if not f.exists() or f.stat().st_size < 500: continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200:
                d[t] = df['Close'].dropna()
        except: pass
    return pd.DataFrame(d)


# ══════════════════════════════════════════════════════════════════════════════
# v11b 日频回测（向量化，移植自 daily_audit_v11b.py）
# ══════════════════════════════════════════════════════════════════════════════
def run_v11b_daily(close_df, sig, sectors,
                   gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
                   start='2015-01-01', end='2025-12-31', cost=0.0015):
    """
    使用 v11b 完整逻辑（含防御桥接、自适应债券、SPY对冲），
    月频调仓，日频净值追踪（向量化）。
    """
    HEDGE_KEYS = v11b.HEDGE_KEYS

    # 构建完整价格矩阵
    hedge_df = pd.DataFrame({
        'GLD': gld_p, 'GDX': gdx_p, 'GDXJ': gdxj_p,
        'SHY': shy_p, 'TLT': tlt_p, 'IEF': ief_p,
    })
    for k, v in def_prices.items():
        hedge_df[k] = v
    all_prices = pd.concat([close_df, hedge_df], axis=1)
    all_prices = all_prices.loc['2014-01-01':end].ffill()
    daily_rets = all_prices.pct_change()

    rng = close_df.loc[start:end].dropna(how='all')
    month_ends = rng.resample('ME').last().index
    SPY_VOL = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None

    # 月频调仓
    periods = []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0
    port_returns = []

    for i in range(len(month_ends) - 1):
        dt, ndt = month_ends[i], month_ends[i + 1]
        dd = (val - peak) / peak if peak > 0 else 0
        spy_vol = float(SPY_VOL.loc[:dt].dropna().iloc[-1]) if (
            SPY_VOL is not None and len(SPY_VOL.loc[:dt].dropna()) > 0) else 0.15
        spy_1m = v11b.get_spy_1m(sig, dt)

        if len(port_returns) >= v11b.VOL_LOOKBACK:
            pv = np.std(port_returns[-v11b.VOL_LOOKBACK:], ddof=1) * np.sqrt(12)
        else:
            pv = 0.20

        w, reg, bond_t = v11b.select(sig, sectors, dt, prev_h,
                                      gld_p, gdx_p, tlt_p, ief_p, def_prices)
        w, shy_boost = v11b.apply_overlays(w, spy_vol, dd, pv, spy_1m)

        all_t = set(w) | set(prev_w)
        to = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2

        invested = sum(w.values())
        cash = max(1.0 - invested, 0.0) + shy_boost
        if cash > 0: w['SHY'] = w.get('SHY', 0) + cash

        periods.append((dt, ndt, dict(w), float(to)))
        prev_w = {k: v for k, v in w.items() if k != 'SHY'}
        prev_h = {k for k in prev_w if k not in HEDGE_KEYS}

        # 月度收益（用于 vol targeting）
        ret = 0.0
        hedge_map = {'GLD': gld_p, 'GDX': gdx_p, 'GDXJ': gdxj_p,
                     'SHY': shy_p, 'TLT': tlt_p, 'IEF': ief_p}
        hedge_map.update(def_prices)
        for t, wt in w.items():
            p = hedge_map.get(t, close_df.get(t))
            if p is None: continue
            s = p.loc[dt:ndt].dropna()
            if len(s) >= 2: ret += (s.iloc[-1] / s.iloc[0] - 1) * wt
        ret -= to * cost * 2
        val *= (1 + ret); peak = max(peak, val)
        port_returns.append(ret)

    # 日频净值（向量化）
    nav_series = [1.0]
    nav_dates  = [month_ends[0]]
    port_val   = 1.0

    for dt, ndt, weights, turnover in periods:
        port_val *= (1.0 - turnover * cost * 2)
        mask = (daily_rets.index > dt) & (daily_rets.index <= ndt)
        period_rets = daily_rets.loc[mask]
        if len(period_rets) == 0: continue

        assets = [t for t in weights if t in daily_rets.columns]
        if not assets:
            for day in period_rets.index:
                nav_series.append(port_val); nav_dates.append(day)
            continue

        w_vec   = np.array([weights.get(t, 0) for t in assets])
        ret_mat = period_rets[assets].fillna(0).values
        port_daily = ret_mat @ w_vec

        for j, day in enumerate(period_rets.index):
            port_val *= (1 + port_daily[j])
            nav_series.append(port_val); nav_dates.append(day)

    nav = pd.Series(nav_series, index=pd.DatetimeIndex(nav_dates))
    nav = nav[~nav.index.duplicated(keep='first')].sort_index().dropna()

    # 裁剪到 start:end
    nav = nav.loc[start:end]
    return nav.rename('StockV11b')


# ══════════════════════════════════════════════════════════════════════════════
# Crypto v3 策略（完整复制自 hybrid_v3_daily.py，加入可配置动量窗口）
# ══════════════════════════════════════════════════════════════════════════════
def run_crypto_v3_daily(btc_p, eth_p, gld_p, shy_p,
                        start='2015-01-01', end='2025-12-31',
                        cost=0.0015, mom_lb=90,
                        ma_window=BTC_MA_WINDOW,
                        dd_exit=BTC_DD_EXIT,
                        dd_reentry=BTC_DD_REENTRY,
                        vol_window=BTC_VOL_WINDOW,
                        vol_target=BTC_VOL_TARGET):
    """Crypto v3：月频动量 + BTC200dMA过滤 + DD保护 + vol targeting"""
    common_idx = btc_p.index.intersection(eth_p.index).intersection(gld_p.index)
    common_idx = pd.DatetimeIndex(common_idx).sort_values()
    common_idx = common_idx[(common_idx >= pd.Timestamp(start)) &
                             (common_idx <= pd.Timestamp(end))]
    if len(common_idx) == 0:
        return pd.Series(dtype=float)

    trading_days = common_idx
    month_ends = pd.Series(index=trading_days).resample('ME').last().index

    # 预计算 BTC 技术指标（全序列）
    btc_log_ret    = np.log(btc_p / btc_p.shift(1))
    btc_200ma      = btc_p.rolling(ma_window, min_periods=int(ma_window*0.8)).mean()
    btc_vol60      = btc_log_ret.rolling(vol_window, min_periods=int(vol_window*0.5)).std() * np.sqrt(252)
    btc_peak_hist  = btc_p.expanding().max()
    btc_dd_series  = (btc_p - btc_peak_hist) / btc_peak_hist

    val = 1.0
    equity_vals, equity_dates = [], []
    monthly_weights = {}
    prev_monthly_w  = {}
    processed_months = set()
    btc_dd_blocked  = False
    vol_scale       = 1.0
    filter_log      = []

    for day_idx, day in enumerate(trading_days):
        # 月频信号
        past_me = month_ends[month_ends < day]
        if len(past_me) > 0:
            last_me = past_me[-1]
            next_days = trading_days[trading_days > last_me]
            exec_day = next_days[0] if len(next_days) > 0 else None

            if exec_day is not None and day == exec_day and last_me not in processed_months:
                btc_hist = btc_p.loc[:last_me].dropna()
                eth_hist = eth_p.loc[:last_me].dropna()

                btc_mom = (btc_hist.iloc[-1] / btc_hist.iloc[-mom_lb] - 1
                           if len(btc_hist) >= mom_lb else -1)
                eth_mom = (eth_hist.iloc[-1] / eth_hist.iloc[-mom_lb] - 1
                           if len(eth_hist) >= mom_lb else -1)

                if max(btc_mom, eth_mom) > 0:
                    new_mw = {'BTC': 0.70, 'ETH': 0.30} if btc_mom >= eth_mom else {'ETH': 0.70, 'BTC': 0.30}
                else:
                    new_mw = {'GLD': 1.00}

                btc_vol_hist = btc_vol60.loc[:last_me].dropna()
                if len(btc_vol_hist) > 0:
                    cur_vol = float(btc_vol_hist.iloc[-1])
                    vol_scale = float(np.clip(vol_target / max(cur_vol, 0.01), BTC_VOL_MIN, BTC_VOL_MAX)) if cur_vol > 0.01 else 1.0
                else:
                    vol_scale = 1.0

                all_t = set(new_mw) | set(prev_monthly_w)
                turnover = sum(abs(new_mw.get(t, 0) - prev_monthly_w.get(t, 0)) for t in all_t) / 2
                val *= (1 - turnover * cost * 2)
                monthly_weights = new_mw.copy()
                prev_monthly_w  = new_mw.copy()
                processed_months.add(last_me)

        # 日频过滤层
        if day in btc_200ma.index and day in btc_dd_series.index:
            btc_price_now = btc_p.loc[day] if day in btc_p.index else np.nan
            btc_ma_now    = btc_200ma.loc[day]
            btc_dd_now    = btc_dd_series.loc[day]

            trend_blocked = bool(pd.notna(btc_ma_now) and pd.notna(btc_price_now) and
                                 btc_price_now < btc_ma_now)
            if not btc_dd_blocked:
                if pd.notna(btc_dd_now) and btc_dd_now < dd_exit:
                    btc_dd_blocked = True
                    filter_log.append({'date': str(day.date()), 'trigger': 'DD_exit',
                                       'btc_dd': round(float(btc_dd_now), 3)})
            else:
                if pd.notna(btc_dd_now) and btc_dd_now > dd_reentry:
                    btc_dd_blocked = False
                    filter_log.append({'date': str(day.date()), 'trigger': 'DD_reentry',
                                       'btc_dd': round(float(btc_dd_now), 3)})
            crypto_blocked = trend_blocked or btc_dd_blocked
        else:
            crypto_blocked = False

        # 构建实际持仓
        if crypto_blocked:
            current_weights = {'GLD': 1.00}
        else:
            cw = {}
            crypto_tickers = [t for t in monthly_weights if t in ('BTC', 'ETH')]
            gld_frac = monthly_weights.get('GLD', 0.0)
            if crypto_tickers and vol_scale != 1.0:
                raw_crypto = sum(monthly_weights.get(t, 0) for t in crypto_tickers)
                adj_crypto = min(raw_crypto * vol_scale, 0.95)
                gld_supplement = gld_frac + max(raw_crypto - adj_crypto, 0)
                sf = adj_crypto / raw_crypto if raw_crypto > 0 else 1.0
                for t in crypto_tickers: cw[t] = monthly_weights[t] * sf
                if gld_supplement > 0.01: cw['GLD'] = gld_supplement
            else:
                cw = monthly_weights.copy()
            current_weights = cw

        # 日频净值
        if day_idx == 0:
            equity_vals.append(val); equity_dates.append(day)
            continue

        prev_day = trading_days[day_idx - 1]
        day_ret = 0.0; invested = 0.0

        for ticker, w in current_weights.items():
            series = {'BTC': btc_p, 'ETH': eth_p, 'GLD': gld_p, 'SHY': shy_p}.get(ticker)
            if series is None: continue
            if prev_day in series.index and day in series.index:
                p0 = series.loc[prev_day]; p1 = series.loc[day]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    day_ret += (p1 / p0 - 1) * w
                    invested += w

        cash_frac = max(1.0 - invested, 0.0)
        if cash_frac > 0.01 and prev_day in shy_p.index and day in shy_p.index:
            sp, st = shy_p.loc[prev_day], shy_p.loc[day]
            if pd.notna(sp) and pd.notna(st) and sp > 0:
                day_ret += (st / sp - 1) * cash_frac

        val *= (1 + day_ret)
        equity_vals.append(val); equity_dates.append(day)

    return pd.Series(equity_vals, index=equity_dates, name=f'CryptoV3_m{mom_lb}')


# ══════════════════════════════════════════════════════════════════════════════
# 指标计算（固定日期 Walk-Forward）
# ══════════════════════════════════════════════════════════════════════════════
def calc_metrics(equity: pd.Series, rf=0.04):
    eq = equity.dropna()
    if len(eq) < 2: return {}
    rets = eq.pct_change().dropna()
    n = len(rets)
    years = n / 252
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    peak = eq.expanding().max()
    dd   = (eq - peak) / peak
    maxdd = float(dd.min())
    ann_ret = float(rets.mean() * 252)
    ann_vol = float(rets.std() * np.sqrt(252))
    sharpe  = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    calmar  = cagr / abs(maxdd) if maxdd != 0 else 0
    composite = sharpe * 0.25 + calmar * 0.25 + min(cagr, 2.0) * 0.50
    return dict(cagr=round(cagr, 4), maxdd=round(maxdd, 4),
                sharpe=round(sharpe, 4), calmar=round(calmar, 4),
                composite=round(composite, 4), ann_vol=round(ann_vol, 4),
                years=round(years, 2))


def walk_forward_fixed(equity: pd.Series,
                       is_end='2020-12-31', oos_start='2021-01-01', rf=0.04):
    eq = equity.dropna()
    is_eq  = eq[eq.index <= pd.Timestamp(is_end)]
    oos_eq = eq[eq.index >= pd.Timestamp(oos_start)]
    if len(is_eq) < 30 or len(oos_eq) < 30:
        return dict(is_sharpe=0, oos_sharpe=0, wf_ratio=0, is_cagr=0, oos_cagr=0)
    is_m  = calc_metrics(is_eq,  rf)
    oos_m = calc_metrics(oos_eq, rf)
    is_sh  = is_m.get('sharpe', 0)
    oos_sh = oos_m.get('sharpe', 0)
    wf = oos_sh / is_sh if is_sh > 0 else 0
    return dict(is_sharpe=round(is_sh, 3), oos_sharpe=round(oos_sh, 3),
                wf_ratio=round(wf, 3),
                is_cagr=round(is_m.get('cagr', 0), 4),
                oos_cagr=round(oos_m.get('cagr', 0), 4))


def build_hybrid(stock_eq: pd.Series, crypto_eq: pd.Series, w_crypto: float):
    """按固定权重合成日频净值"""
    common = stock_eq.index.intersection(crypto_eq.index).sort_values()
    s_ret = stock_eq.loc[common].pct_change().fillna(0)
    c_ret = crypto_eq.loc[common].pct_change().fillna(0)
    port_ret = (1 - w_crypto) * s_ret + w_crypto * c_ret
    port_vals = (1 + port_ret).cumprod()
    port_vals.iloc[0] = 1.0
    return pd.Series(port_vals.values, index=common, name=f'HybV4_c{int(w_crypto*100)}')


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("Hybrid v4 Daily — Stock v11b + Crypto v3 + Direction C 🐻")
    print("股票升级(v9j→v11b) + BTC200dMA过滤 + 45d短动量测试")
    print("=" * 80)

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    print("\n[1/6] 加载市场数据...")
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    if 'SPY' not in close_df.columns:
        raise ValueError("SPY 数据缺失！")
    print(f"  ✓ 股票：{len(close_df.columns)} 只，{len(close_df)} 日")

    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    tlt_p  = load_csv(CACHE / "TLT.csv")['Close'].dropna()

    ief_path = CACHE / "IEF.csv"
    ief_p = load_csv(ief_path)['Close'].dropna() if ief_path.exists() else shy_p.copy()

    # 防御 ETF（v11b 新增）
    def_prices = {}
    for etf in ['XLV', 'XLP', 'XLU']:
        p = CACHE / f"{etf}.csv"
        if p.exists():
            def_prices[etf] = load_csv(p)['Close'].dropna()
        else:
            # 用 SPY 替代（保守处理）
            def_prices[etf] = close_df['SPY'].copy()
    print(f"  ✓ 防御ETF：{list(def_prices.keys())} (v11b 桥接层)")

    btc_p = load_csv(CACHE / "BTC_USD.csv")['Close'].dropna()
    eth_p = load_csv(CACHE / "ETH_USD.csv")['Close'].dropna()
    print(f"  ✓ Crypto：BTC={len(btc_p)}d, ETH={len(eth_p)}d")

    # ── 预计算 v11b 信号 ──────────────────────────────────────────────────────
    print("[2/6] 预计算 v11b 股票信号...")
    sig     = v11b.precompute(close_df)
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    print(f"  ✓ 行业分类：{len(sectors)} 只")

    # ── 运行 v11b 日频回测 ────────────────────────────────────────────────────
    print("[3/6] 运行 Stock v11b 日频回测...")
    stock_eq = run_v11b_daily(
        close_df, sig, sectors,
        gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
        start='2015-01-01', end='2025-12-31'
    )
    stock_m  = calc_metrics(stock_eq)
    stock_wf = walk_forward_fixed(stock_eq)
    print(f"  ✓ Stock v11b：CAGR={stock_m['cagr']:.1%}  MaxDD={stock_m['maxdd']:.1%}  "
          f"Sharpe={stock_m['sharpe']:.3f}  WF={stock_wf['wf_ratio']:.3f}")
    print(f"    IS CAGR={stock_wf['is_cagr']:.1%}  OOS CAGR={stock_wf['oos_cagr']:.1%}  "
          f"IS_Sh={stock_wf['is_sharpe']:.3f}  OOS_Sh={stock_wf['oos_sharpe']:.3f}")

    # ── 运行 Crypto v3（90d 动量）────────────────────────────────────────────
    print("[4/6] 运行 Crypto v3（90d 动量，三重过滤）...")
    crypto_90d = run_crypto_v3_daily(btc_p, eth_p, gld_p, shy_p,
                                      start='2015-01-01', end='2025-12-31',
                                      mom_lb=90)
    c90_m  = calc_metrics(crypto_90d)
    c90_wf = walk_forward_fixed(crypto_90d)
    print(f"  ✓ Crypto v3(90d)：CAGR={c90_m['cagr']:.1%}  MaxDD={c90_m['maxdd']:.1%}  "
          f"Sharpe={c90_m['sharpe']:.3f}  WF={c90_wf['wf_ratio']:.3f}")

    # ── 运行 Crypto v3b（45d 动量，Direction C）──────────────────────────────
    print("[5/6] 运行 Crypto v3b（45d 短动量，Direction C）...")
    crypto_45d = run_crypto_v3_daily(btc_p, eth_p, gld_p, shy_p,
                                      start='2015-01-01', end='2025-12-31',
                                      mom_lb=45)
    c45_m  = calc_metrics(crypto_45d)
    c45_wf = walk_forward_fixed(crypto_45d)
    print(f"  ✓ Crypto v3b(45d)：CAGR={c45_m['cagr']:.1%}  MaxDD={c45_m['maxdd']:.1%}  "
          f"Sharpe={c45_m['sharpe']:.3f}  WF={c45_wf['wf_ratio']:.3f}")
    print(f"    vs 90d：CAGR {c90_m['cagr']:.1%}→{c45_m['cagr']:.1%}  "
          f"MaxDD {c90_m['maxdd']:.1%}→{c45_m['maxdd']:.1%}  "
          f"WF {c90_wf['wf_ratio']:.3f}→{c45_wf['wf_ratio']:.3f}")

    # ── 混合组合扫描 ──────────────────────────────────────────────────────────
    print("\n[6/6] 混合组合扫描...")
    w_list = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40]

    configs = [
        ('v11b+C90', stock_eq, crypto_90d),
        ('v11b+C45', stock_eq, crypto_45d),
    ]

    all_results = {}
    for (name, seq, ceq) in configs:
        results = []
        for w_c in w_list:
            if w_c == 0.0:
                eq = seq
            else:
                eq = build_hybrid(seq, ceq, w_c)
            m  = calc_metrics(eq)
            wf = walk_forward_fixed(eq)
            row = dict(w_crypto=w_c, config=name, **m, **wf)
            results.append(row)
        all_results[name] = results

    # ── 结果打印 ──────────────────────────────────────────────────────────────
    print()
    print("=" * 120)
    print(f"{'配置':>12} {'w_c':>5} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} "
          f"{'Calmar':>8} {'Composite':>10} {'WF':>7} "
          f"{'IS_Sh':>7} {'OOS_Sh':>7} {'IS_CAG':>8} {'OOS_CAG':>9}")
    print("-" * 120)

    best_overall = None
    best_comp = -99

    for name, results in all_results.items():
        for r in results:
            wf_flag = "✅" if r['wf_ratio'] >= 0.60 else "❌"
            wf_str  = f"{wf_flag}{r['wf_ratio']:.2f}"
            extra   = ""
            if r['wf_ratio'] >= 0.60 and r['composite'] > best_comp:
                best_comp    = r['composite']
                best_overall = r
            if r['w_crypto'] == 0.0:
                extra = " ← v11b 纯股票基准"
            print(f"{r['config']:>12}  {r['w_crypto']:>4.0%}  "
                  f"{r['cagr']:>7.1%}  {r['maxdd']:>7.1%}  "
                  f"{r['sharpe']:>7.3f}  {r['calmar']:>7.3f}  "
                  f"{r['composite']:>9.3f}  {wf_str:>7}  "
                  f"{r.get('is_sharpe',0):>6.3f}  {r.get('oos_sharpe',0):>6.3f}  "
                  f"{r.get('is_cagr',0):>7.1%}  {r.get('oos_cagr',0):>8.1%}{extra}")
        print()

    print("=" * 120)

    # ── 对比汇总 ──────────────────────────────────────────────────────────────
    print()
    print("── 关键对比 ────────────────────────────────────────────────────────────")

    # v9j + Crypto v3 @ 20% (Hybrid v3 最优)
    v3_ref = dict(config='HybV3(v9j+C90)', w_crypto=0.20,
                  cagr=0.527, maxdd=-0.220, sharpe=2.096,
                  calmar=2.392, composite=1.386, wf_ratio=0.600,
                  is_sharpe=2.550, oos_sharpe=1.531, is_cagr=0.441, oos_cagr=0.352)

    print(f"{'HybV3(v9j+C90)':>12}  {'20%':>4}  {v3_ref['cagr']:>7.1%}  "
          f"{v3_ref['maxdd']:>7.1%}  {v3_ref['sharpe']:>7.3f}  "
          f"{v3_ref['calmar']:>7.3f}  {v3_ref['composite']:>9.3f}  "
          f"✅{v3_ref['wf_ratio']:.2f}   "
          f"{v3_ref['is_sharpe']:>6.3f}  {v3_ref['oos_sharpe']:>6.3f}  "
          f"{v3_ref['is_cagr']:>7.1%}  {v3_ref['oos_cagr']:>8.1%}  ← v3 冠军(参考)")

    if best_overall:
        cagr_vs_v3 = best_overall['cagr'] - v3_ref['cagr']
        wf_vs_v3   = best_overall['wf_ratio'] - v3_ref['wf_ratio']
        dd_vs_v3   = best_overall['maxdd'] - v3_ref['maxdd']
        comp_vs_v3 = best_overall['composite'] - v3_ref['composite']

        print()
        print(f"🏆 v4 最优（WF≥0.6）：{best_overall['config']}  w_crypto={best_overall['w_crypto']:.0%}")
        print(f"   CAGR={best_overall['cagr']:.1%} / MaxDD={best_overall['maxdd']:.1%} / "
              f"WF={best_overall['wf_ratio']:.3f} / Sharpe={best_overall['sharpe']:.3f} / "
              f"Composite={best_overall['composite']:.3f}")
        print(f"   vs Hybrid v3 最优：CAGR {v3_ref['cagr']:.1%}→{best_overall['cagr']:.1%} "
              f"({'+'if cagr_vs_v3>=0 else ''}{cagr_vs_v3:.1%})  "
              f"WF {'+'if wf_vs_v3>=0 else ''}{wf_vs_v3:.3f}  "
              f"MaxDD {'+'if dd_vs_v3>=0 else ''}{dd_vs_v3:.1%}")

        if (best_overall['cagr'] > 0.40 and best_overall['wf_ratio'] >= 0.60
                and best_overall['maxdd'] > -0.30):
            print("\n🚀🚀🚀 【重大突破】 CAGR>40% + WF≥0.60 + MaxDD<30% ！")
            if best_overall['cagr'] > v3_ref['cagr'] + 0.05:
                print("🌟 且相比 v3 CAGR 提升超过 5%！")
        else:
            print("\n⚠️  未超越 v3 突破")

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent

    save_data = {
        'stock_v11b': {**stock_m, **stock_wf},
        'crypto_v3_90d': {**c90_m, **c90_wf},
        'crypto_v3_45d': {**c45_m, **c45_wf},
        'hybrid_v4_sweep': {k: v for k, v in all_results.items()},
        'best_overall': best_overall,
        'v3_reference': v3_ref,
        'meta': {
            'date':   '2026-02-22',
            'author': '代码熊 🐻',
            'desc':   'Hybrid v4: v11b股票(Sharpe=1.96) + Crypto v3(200dMA+DD+vol) + Direction C(45d)',
            'wf_is':  '2015-01-01 ~ 2020-12-31',
            'wf_oos': '2021-01-01 ~ 2025-12-31',
            'composite_formula': 'Sharpe*0.25 + Calmar*0.25 + min(CAGR,2.0)*0.50',
        }
    }

    results_path = out_dir / "hybrid_v4_results.json"
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n📁 结果：{results_path}")

    # 净值曲线
    eq_df = pd.DataFrame({
        'StockV11b':    stock_eq,
        'CryptoV3_90d': crypto_90d,
        'CryptoV3_45d': crypto_45d,
    }).dropna()
    if best_overall and best_overall['w_crypto'] > 0:
        name = best_overall['config']
        ceq = crypto_90d if '90' in name else crypto_45d
        best_eq = build_hybrid(stock_eq, ceq, best_overall['w_crypto'])
        eq_df['HybridV4_Best'] = best_eq
    eq_df.to_csv(out_dir / "hybrid_v4_equity.csv")
    print(f"📁 净值曲线：{out_dir / 'hybrid_v4_equity.csv'}")

    return all_results, best_overall


if __name__ == '__main__':
    results, best = main()
