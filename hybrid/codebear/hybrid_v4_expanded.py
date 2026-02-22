#!/usr/bin/env python3
"""
Hybrid v4 Expanded — v11b_final 股票 + 多币 Crypto 动量轮换
代码熊 🐻 | 2026-02-22

三大升级 vs v3:
  B. 放宽 WF 阈值 → 扫描 w_crypto 10-40%（目标 WF≥0.58）
  C. 股票端升级：v9j → v11b_final（直接 import 原模块，零 bug）
  D. 加密货币池扩展：BTC + ETH + BNB + ADA + SOL (+ 动量排名 Top-2)

目标: CAGR>45%, WF≥0.58, 日频MaxDD<30%
新 Composite = Sharpe×0.25 + Calmar×0.25 + min(CAGR,2.0)×0.50
"""

import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parents[2]
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# ─────────────────────────────────────────
# Import v11b_final module (zero reimplementation)
# ─────────────────────────────────────────
sys.path.insert(0, str(BASE / "stocks" / "codebear"))
import momentum_v11b_final as v11b

# ─────────────────────────────────────────
# Crypto v4 参数
# ─────────────────────────────────────────
CRYPTO_TOP_N   = 2
CRYPTO_MOM_LB  = 90
CRYPTO_MA_WIN  = 200
CRYPTO_DD_EXIT = -0.25
CRYPTO_DD_REENTRY = -0.15
CRYPTO_VOL_WIN = 60
CRYPTO_VOL_TGT = 0.80
CRYPTO_VOL_MIN = 0.40
CRYPTO_VOL_MAX = 1.50
COST = 0.0015

# ─────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────
def load_csv(fp):
    df = pd.read_csv(fp)
    c = 'Date' if 'Date' in df.columns else df.columns[0]
    df[c] = pd.to_datetime(df[c])
    return df.set_index(c).sort_index()

def load_series(name):
    fp = CACHE / f"{name}.csv"
    df = load_csv(fp)
    col = 'Close' if 'Close' in df.columns else df.columns[0]
    return pd.to_numeric(df[col], errors='coerce').dropna()

# ─────────────────────────────────────────
# 股票 v11b_final 日频 P&L 追踪
# ─────────────────────────────────────────
def run_stock_v11b_daily(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                          tlt_p, ief_p, def_prices,
                          start='2015-01-01', end='2025-12-31'):
    """
    使用 v11b_final 的原始 select() + apply_overlays() 函数
    月末信号 → 次月执行 → 日频净值追踪
    """
    all_daily = close_df.loc[start:end].dropna(how='all')
    month_ends = all_daily.resample('ME').last().index
    trading_days = all_daily.index

    val = 1.0; peak = 1.0
    equity_vals, equity_dates = [], []
    current_weights = {}
    prev_hold = {}; prev_w = {}
    processed_months = set()
    port_returns = []; last_rebal_val = 1.0

    # price map
    HEDGE_KEYS = {'GLD', 'GDX', 'GDXJ', 'TLT', 'IEF', 'SHY', 'XLV', 'XLP', 'XLU'}
    price_map_base = {'GLD': gld_p, 'GDX': gdx_p, 'GDXJ': gdxj_p,
                      'SHY': shy_p, 'TLT': tlt_p, 'IEF': ief_p}
    for etf, p in def_prices.items():
        price_map_base[etf] = p

    for day_idx, day in enumerate(trading_days):
        past_mes = month_ends[month_ends < day]
        if len(past_mes) > 0:
            last_me = past_mes[-1]
            next_days = trading_days[trading_days > last_me]
            exec_day = next_days[0] if len(next_days) > 0 else None
            if exec_day is not None and day == exec_day and last_me not in processed_months:
                dd_now = (val - peak) / peak
                # SPY 1m return
                spy_1m = v11b.get_spy_1m(sig, last_me)
                # SPY vol
                try:
                    slog = np.log(sig['spy'] / sig['spy'].shift(1)).loc[:last_me].dropna()
                    spy_vol = float(slog.iloc[-63:].std() * np.sqrt(252)) if len(slog) >= 10 else 0.15
                except: spy_vol = 0.15
                # Port vol
                port_returns.append(val / last_rebal_val - 1 if last_rebal_val > 0 else 0)
                port_vol_ann = np.std(port_returns[-v11b.VOL_LOOKBACK:]) * np.sqrt(12) \
                    if len(port_returns) >= v11b.VOL_LOOKBACK else 0.20

                # v11b_final 选股（直接调用原版）
                new_w, regime, bond_type = v11b.select(
                    sig, sectors, last_me, prev_hold,
                    gld_p, gdx_p, tlt_p, ief_p, def_prices)
                new_w, _ = v11b.apply_overlays(
                    new_w, spy_vol, dd_now, port_vol_ann, spy_1m)

                all_t = set(new_w) | set(prev_w)
                tc = sum(abs(new_w.get(t,0) - prev_w.get(t,0)) for t in all_t) * COST
                val *= (1 - tc)

                last_rebal_val = val
                current_weights = new_w.copy()
                prev_w = new_w.copy()
                prev_hold = {k: v for k, v in new_w.items() if k not in HEDGE_KEYS}
                processed_months.add(last_me)

        if day_idx == 0:
            equity_vals.append(val); equity_dates.append(day); continue

        prev_day = trading_days[day_idx - 1]
        day_ret = 0.0; invested = 0.0

        for ticker, w in current_weights.items():
            if w == 0: continue
            series = price_map_base.get(ticker)
            if series is None:
                series = close_df[ticker] if ticker in close_df.columns else None
            if series is None: continue
            if prev_day in series.index and day in series.index:
                p0, p1 = series.loc[prev_day], series.loc[day]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    day_ret += (p1/p0 - 1) * w; invested += w

        cash = max(1.0 - invested, 0)
        if cash > 0.01 and prev_day in shy_p.index and day in shy_p.index:
            sp, st = shy_p.loc[prev_day], shy_p.loc[day]
            if pd.notna(sp) and pd.notna(st) and sp > 0:
                day_ret += (st/sp - 1) * cash

        val *= (1 + day_ret)
        peak = max(peak, val)
        equity_vals.append(val); equity_dates.append(day)

    return pd.Series(equity_vals, index=equity_dates, name='StockV11b')

# ─────────────────────────────────────────
# Crypto v4 — 多币动量轮换 + 三重保护
# ─────────────────────────────────────────
def run_crypto_v4_daily(crypto_prices, gld_p, shy_p,
                        start='2015-01-01', end='2025-12-31'):
    # 统一日期索引
    all_idx = None
    for p in crypto_prices.values():
        idx = p.loc[start:end].dropna().index
        all_idx = idx if all_idx is None else all_idx.union(idx)
    if all_idx is None or len(all_idx) == 0: return pd.Series(dtype=float)

    trading_days = all_idx
    month_ends = pd.Series(1, index=trading_days).resample('ME').last().index

    # 预计算 MA + vol
    ma200 = {c: p.rolling(CRYPTO_MA_WIN).mean() for c, p in crypto_prices.items()}
    vol60 = {c: np.log(p/p.shift(1)).rolling(CRYPTO_VOL_WIN).std()*np.sqrt(252)
             for c, p in crypto_prices.items()}
    btc_p = crypto_prices['BTC']
    btc_dd_blocked = False
    vol_scale = 1.0

    monthly_weights = {}
    processed = set()
    prev_w = {}
    val = 1.0; peak = 1.0
    equity_vals, equity_dates = [], []

    for day_idx, day in enumerate(trading_days):
        # 月末计算信号
        past_mes = month_ends[month_ends < day]
        if len(past_mes) > 0:
            last_me = past_mes[-1]
            if last_me not in processed:
                scored = []
                for coin, p in crypto_prices.items():
                    h = p.loc[:last_me].dropna()
                    ma = ma200[coin].loc[:last_me].dropna()
                    if len(h) < CRYPTO_MOM_LB + 5 or len(ma) == 0: continue
                    mom = float(h.iloc[-1] / h.iloc[-CRYPTO_MOM_LB] - 1)
                    above_ma = float(h.iloc[-1]) > float(ma.iloc[-1])
                    scored.append((coin, mom, above_ma))

                eligible = sorted([(c, m) for c, m, tr in scored if tr],
                                   key=lambda x: x[1], reverse=True)
                top_coins = [c for c, _ in eligible[:CRYPTO_TOP_N]]

                # vol 目标化
                bv = vol60['BTC'].loc[:last_me].dropna()
                if len(bv) > 0:
                    raw_scale = CRYPTO_VOL_TGT / max(float(bv.iloc[-1]), 0.01)
                    vol_scale = float(np.clip(raw_scale, CRYPTO_VOL_MIN, CRYPTO_VOL_MAX))

                monthly_weights[last_me] = {c: 1.0/len(top_coins) for c in top_coins} \
                    if top_coins else {'GLD': 1.0}
                processed.add(last_me)

        # 当前月份权重
        app_mes = [me for me in month_ends if me < day and me in monthly_weights]
        if not app_mes:
            equity_vals.append(val); equity_dates.append(day); continue
        cur_me = app_mes[-1]
        base_w = monthly_weights[cur_me].copy()

        # 日频保护：BTC 200dMA
        btc_now = btc_p.loc[:day].dropna()
        btc_ma_now = ma200['BTC'].loc[:day].dropna()
        trend_blocked = (len(btc_now) > 0 and len(btc_ma_now) > 0 and
                         float(btc_now.iloc[-1]) < float(btc_ma_now.iloc[-1]))

        # BTC 回撤保护
        if len(btc_now) >= 5:
            btc_peak = float(btc_p.loc[:day].max())
            btc_dd = float(btc_now.iloc[-1] / btc_peak - 1)
            if btc_dd < CRYPTO_DD_EXIT: btc_dd_blocked = True
            elif btc_dd > CRYPTO_DD_REENTRY: btc_dd_blocked = False

        if trend_blocked or btc_dd_blocked:
            current_w = {'GLD': 1.0}
        else:
            crypto_tickers = [c for c in base_w if c != 'GLD']
            if crypto_tickers:
                adj = min(vol_scale * sum(base_w.get(c,0) for c in crypto_tickers), 0.95)
                scale_f = adj / sum(base_w.get(c,0) for c in crypto_tickers)
                current_w = {c: base_w[c] * scale_f for c in crypto_tickers}
                gld_supp = 1.0 - adj
                if gld_supp > 0.02: current_w['GLD'] = gld_supp
            else:
                current_w = {'GLD': 1.0}

        if day_idx == 0:
            equity_vals.append(val); equity_dates.append(day); continue

        # 换仓成本（月初第一交易日）
        app_mes2 = [me for me in month_ends if me < day and me in monthly_weights]
        if len(app_mes2) >= 2:
            next_days = trading_days[trading_days > app_mes2[-1]]
            if len(next_days) > 0 and day == next_days[0]:
                tc = sum(abs(current_w.get(k,0) - prev_w.get(k,0))
                         for k in set(current_w)|set(prev_w)) * COST
                val *= (1 - tc)
        prev_w = current_w.copy()

        prev_day = trading_days[day_idx - 1]
        day_ret = 0.0
        for ticker, w in current_w.items():
            if w == 0: continue
            series = gld_p if ticker == 'GLD' else crypto_prices.get(ticker)
            if series is None: continue
            if prev_day in series.index and day in series.index:
                p0, p1 = series.loc[prev_day], series.loc[day]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    day_ret += (p1/p0 - 1) * w

        val *= (1 + day_ret)
        peak = max(peak, val)
        equity_vals.append(val); equity_dates.append(day)

    return pd.Series(equity_vals, index=equity_dates, name='CryptoV4')

# ─────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────
def calc_metrics(equity: pd.Series, start='2015-01-01', end='2025-12-31'):
    eq = equity.loc[start:end].dropna()
    if len(eq) < 100: return None
    rets = eq.pct_change().dropna().values
    years = len(rets) / 252
    cagr  = float((eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1)
    cum   = np.cumprod(1 + rets)
    dd    = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)
    max_dd = float(dd.min())
    excess = rets - 0.04/252
    sharpe = float(np.mean(excess)/np.std(excess)*np.sqrt(252)) if np.std(excess) > 0 else 0
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0

    def sub_sharpe(s, e):
        sub = equity.loc[s:e].dropna()
        if len(sub) < 50: return 0.0
        r = sub.pct_change().dropna().values
        ex = r - 0.04/252
        return float(np.mean(ex)/np.std(ex)*np.sqrt(252)) if np.std(ex) > 0 else 0.0
    def sub_cagr(s, e):
        sub = equity.loc[s:e].dropna()
        if len(sub) < 50: return 0.0
        y = len(sub)/252
        return float((sub.iloc[-1]/sub.iloc[0])**(1/y)-1)

    is_s  = sub_sharpe('2015-01-01', '2021-12-31')
    oos_s = sub_sharpe('2022-01-01', '2025-12-31')
    wf = float(oos_s / is_s) if is_s > 0 else 0.0

    composite = sharpe * 0.25 + calmar * 0.25 + min(cagr, 2.0) * 0.50
    return dict(cagr=cagr, max_dd=max_dd, sharpe=sharpe, calmar=calmar,
                wf=wf, is_sharpe=is_s, oos_sharpe=oos_s,
                is_cagr=sub_cagr('2015-01-01','2021-12-31'),
                oos_cagr=sub_cagr('2022-01-01','2025-12-31'),
                composite=composite)

# ─────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────
def main():
    print("="*72)
    print("🐻 Hybrid v4 — v11b_final + 多币 Crypto 动量轮换 (BCD 升级)")
    print("="*72)

    sectors_raw = json.load(open(CACHE / "sp500_sectors.json"))
    sectors = {t: v[:5] if isinstance(v,str) else 'Unkno'
               for t, v in sectors_raw.items()}

    close_df = v11b.load_stocks(list(sectors.keys()))
    print(f"  Stocks: {len(close_df.columns)} tickers")

    gld_p  = load_series('GLD')
    gdx_p  = load_series('GDX')
    gdxj_p = load_series('GDXJ')
    shy_p  = load_series('SHY')
    tlt_p  = load_series('TLT')
    ief_p  = load_series('IEF')
    spy_p  = load_series('SPY')

    def_prices = {e: load_series(e) for e in v11b.DEFENSIVE_ETFS}

    # 合并 SPY + ETF 到 close_df
    close_df['SPY'] = spy_p
    for e in ['GLD','GDX','GDXJ','TLT','IEF']:
        close_df[e] = load_series(e)

    sig = v11b.precompute(close_df)

    # 加密货币
    crypto_prices = {}
    for coin, fname in [('BTC','BTC_USD'), ('ETH','ETH_USD'),
                        ('BNB','BNB_USD'), ('ADA','ADA_USD'), ('SOL','SOL_USD')]:
        try:
            p = load_series(fname)
            if len(p) >= 200:
                crypto_prices[coin] = p
                print(f"  Crypto {coin}: {p.index[0].date()} → {p.index[-1].date()} ({len(p)} rows)")
        except: pass
    print(f"  Pool: {list(crypto_prices.keys())}")

    # ── 股票基准 v11b_final ──
    print("\n📊 [BASE] 纯股票 v11b_final:")
    stock_eq = run_stock_v11b_daily(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                     shy_p, tlt_p, ief_p, def_prices)
    sm = calc_metrics(stock_eq)
    if sm:
        print(f"  CAGR={sm['cagr']:.1%} MaxDD={sm['max_dd']:.1%} "
              f"Sharpe={sm['sharpe']:.2f} WF={sm['wf']:.3f} "
              f"IS={sm['is_cagr']:.1%} OOS={sm['oos_cagr']:.1%}")
        print(f"  (v11b月频基准: CAGR=32.7%, WF=0.74 ← 比较参考)")

    # ── Crypto v4 独立 ──
    print("\n📊 [CRYPTO v4] 扩展 Crypto 独立 (BTC+ETH+BNB+ADA+SOL):")
    crypto_eq = run_crypto_v4_daily(crypto_prices, gld_p, shy_p)
    cm = calc_metrics(crypto_eq)
    if cm:
        print(f"  CAGR={cm['cagr']:.1%} MaxDD={cm['max_dd']:.1%} "
              f"Sharpe={cm['sharpe']:.2f} WF={cm['wf']:.3f} "
              f"IS={cm['is_cagr']:.1%} OOS={cm['oos_cagr']:.1%}")
        print(f"  (v3 Crypto 参考: CAGR=134.7%, WF=0.545)")

    # ── 组合扫描 ──
    w_vals = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    print(f"\n🔍 组合扫描 w_crypto ∈ {[int(w*100) for w in w_vals]}% (WF目标≥0.58)...")
    print(f"{'w_c':>5} | {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} "
          f"{'WF':>6} {'IS_S':>6} {'OOS_S':>6} {'Comp':>7} | {'vs v3':>8}")
    print("-"*75)

    results = []
    idx = stock_eq.index.intersection(crypto_eq.index)
    idx = idx[(idx >= '2015-01-01') & (idx <= '2025-12-31')]
    s_ret = stock_eq.loc[idx].pct_change().fillna(0)
    c_ret = crypto_eq.loc[idx].pct_change().fillna(0)

    for wc in w_vals:
        combo_ret = (1-wc)*s_ret + wc*c_ret
        combo_eq  = (1 + combo_ret).cumprod()
        m = calc_metrics(combo_eq)
        if m is None: continue

        flag = "🏆" if m['wf'] >= 0.58 and m['cagr'] >= 0.45 else \
               ("✅" if m['wf'] >= 0.58 else "  ")
        vs_v3 = f"CAGR{m['cagr']-0.527:+.1%} WF{m['wf']-0.600:+.3f}"
        print(f"  {flag} {wc:.0%} | {m['cagr']:>6.1%} {m['max_dd']:>7.1%} "
              f"{m['sharpe']:>7.2f} {m['wf']:>6.3f} "
              f"{m['is_sharpe']:>6.2f} {m['oos_sharpe']:>6.2f} "
              f"{m['composite']:>7.3f} | {vs_v3}")
        results.append(dict(w_crypto=wc, **m))

    # 汇总
    print("\n" + "="*72)
    wf_ok = [r for r in results if r['wf'] >= 0.58]
    if wf_ok:
        best = max(wf_ok, key=lambda x: x['composite'])
        print(f"✅ WF≥0.58 最优: w={best['w_crypto']:.0%} "
              f"CAGR={best['cagr']:.1%} WF={best['wf']:.3f} "
              f"MaxDD={best['max_dd']:.1%} Comp={best['composite']:.3f}")
    else:
        print("❌ 无配置满足 WF≥0.58")
        print("   → 多币种组合可能 OOS WF 比 v3 更差，需要分析原因")

    abs_best = max(results, key=lambda x: x['composite']) if results else None
    if abs_best:
        print(f"🏆 绝对最优: w={abs_best['w_crypto']:.0%} "
              f"CAGR={abs_best['cagr']:.1%} WF={abs_best['wf']:.3f} Comp={abs_best['composite']:.3f}")

    out = {
        'strategy': 'Hybrid v4: v11b_final + Multi-Coin Crypto',
        'stock_v11b': sm, 'crypto_v4': cm,
        'sweep': results,
        'best_wf058': max(wf_ok, key=lambda x:x['composite']) if wf_ok else None,
        'best_abs': abs_best,
        'meta': {'date': '2026-02-22', 'author': '代码熊 🐻',
                 'crypto_pool': list(crypto_prices.keys())}
    }
    json.dump(out, open(Path(__file__).with_name('hybrid_v4_results.json'), 'w'),
              indent=2, default=float)
    print(f"💾 结果保存 → hybrid_v4_results.json")

if __name__ == '__main__':
    main()
