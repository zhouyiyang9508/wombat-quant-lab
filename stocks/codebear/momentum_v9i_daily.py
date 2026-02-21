#!/usr/bin/env python3
"""
动量轮动 v9i-Daily — 日频净值追踪版本（波动率目标化）
代码熊 🐻

基于 v9i_final 的选股逻辑，改为月频调仓 + 日频净值追踪
核心：波动率目标化 scale 仍然每月末更新（使用月度历史收益率计算）

📋 审计目的：
  揭示真实日频MaxDD，对比月频回测报告的-10.7%是否低估

注意事项：
  - port_returns 积累月度持仓收益（信号日→下个月信号日之间的收益）
  - scale 每月末更新，NOT 每日更新
  - 日频只用于计算净值曲线，不改变月末调仓逻辑

严格前瞻检查（已通过，详见 v9i_audit_report.md）:
  - Vol targeting 使用历史月度收益 ✅
  - Breadth 使用 .loc[:date] ✅
  - GLD/GDXJ 6m动量无前瞻 ✅
  - 动态行业集中度用当月末 breadth 决定下月仓位 ✅
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# ── Champion parameters (identical to v9i_final) ────────────────────────────
MOM_W          = (0.20, 0.50, 0.20, 0.10)
N_BULL_SECS    = 5; N_BULL_SECS_HI = 4; BREADTH_CONC = 0.65
BULL_SPS = 2;  BEAR_SPS = 2; BREADTH_NARROW = 0.45
GLD_AVG_THRESH = 0.70; GLD_COMPETE_FRAC = 0.20
GDX_AVG_THRESH = 0.20; GDX_COMPETE_FRAC = 0.04
CONT_BONUS = 0.03; HI52_FRAC = 0.60; USE_SHY = True
DD_PARAMS = {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}
GDXJ_VOL_LO_THRESH = 0.30; GDXJ_VOL_LO_FRAC = 0.08
GDXJ_VOL_HI_THRESH = 0.45; GDXJ_VOL_HI_FRAC = 0.18
VOL_TARGET_ANN = 0.11
VOL_LOOKBACK   = 3


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


def precompute(close_df):
    r1   = close_df / close_df.shift(22)  - 1
    r3   = close_df / close_df.shift(63)  - 1
    r6   = close_df / close_df.shift(126) - 1
    r12  = close_df / close_df.shift(252) - 1
    r52w_hi = close_df.rolling(252).max()
    log_r = np.log(close_df / close_df.shift(1))
    vol5  = log_r.rolling(5).std() * np.sqrt(252)
    vol30 = log_r.rolling(30).std() * np.sqrt(252)
    spy   = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200  = spy.rolling(200).mean() if spy is not None else None
    sma50 = close_df.rolling(50).mean()
    return dict(r1=r1, r3=r3, r6=r6, r12=r12, r52w_hi=r52w_hi,
                vol5=vol5, vol30=vol30, spy=spy, s200=s200,
                sma50=sma50, close=close_df)


def get_spy_vol(sig, date):
    if sig['vol5'] is None or 'SPY' not in sig['vol5'].columns: return 0.15
    v = sig['vol5']['SPY'].loc[:date].dropna()
    return float(v.iloc[-1]) if len(v) > 0 else 0.15


def compute_breadth(sig, date):
    close = sig['close'].loc[:date].dropna(how='all')
    sma50 = sig['sma50'].loc[:date].dropna(how='all')
    if len(close) < 50: return 1.0
    lc = close.iloc[-1]; ls = sma50.iloc[-1]
    mask = (lc > ls).dropna()
    return float(mask.sum() / len(mask)) if len(mask) > 0 else 1.0


def get_regime(sig, date):
    if sig['s200'] is None: return 'bull'
    spy_now  = sig['spy'].loc[:date].dropna()
    s200_now = sig['s200'].loc[:date].dropna()
    if len(spy_now) == 0 or len(s200_now) == 0: return 'bull'
    breadth = compute_breadth(sig, date)
    return 'bear' if (spy_now.iloc[-1] < s200_now.iloc[-1] and
                      breadth < BREADTH_NARROW) else 'bull'


def asset_compete(sig, date, prices, thresh, frac, lb=127):
    r6  = sig['r6']
    idx = r6.index[r6.index <= date]
    if len(idx) < 1: return 0.0
    d = idx[-1]
    stock_r6 = r6.loc[d].dropna()
    stock_r6 = stock_r6[stock_r6 > 0]
    if len(stock_r6) < 10: return 0.0
    avg_r6 = stock_r6.mean()
    hist = prices.loc[:d].dropna()
    if len(hist) < lb + 3: return 0.0
    asset_r6 = hist.iloc[-1] / hist.iloc[-lb] - 1
    return frac if asset_r6 >= avg_r6 * thresh else 0.0


def select(sig, sectors, date, prev_hold, gld_p, gdx_p):
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}
    d = idx[-1]

    w1, w3, w6, w12 = MOM_W
    mom = (sig['r1'].loc[d]*w1 + sig['r3'].loc[d]*w3 +
           sig['r6'].loc[d]*w6 + sig['r12'].loc[d]*w12)

    df = pd.DataFrame({
        'mom':   mom, 'r6':  sig['r6'].loc[d],
        'vol':   sig['vol30'].loc[d], 'price': close.loc[d],
        'sma50': sig['sma50'].loc[d], 'hi52':  sig['r52w_hi'].loc[d],
    }).dropna(subset=['mom', 'sma50'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
    df = df[df['price'] > df['sma50']]
    df = df[df['price'] >= df['hi52'] * HI52_FRAC]
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    df = df[df['sector'] != 'Unknown']
    for t in df.index:
        if t in prev_hold: df.loc[t, 'mom'] += CONT_BONUS
    if len(df) == 0: return {}

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)
    gld_a = asset_compete(sig, date, gld_p, GLD_AVG_THRESH, GLD_COMPETE_FRAC)
    gdx_a = asset_compete(sig, date, gdx_p, GDX_AVG_THRESH, GDX_COMPETE_FRAC)
    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)
    reg     = get_regime(sig, date)
    breadth = compute_breadth(sig, date)

    if reg == 'bull':
        n_bull = N_BULL_SECS_HI if breadth > BREADTH_CONC else N_BULL_SECS
        n_secs = max(n_bull - n_compete, 1)
        sps, cash = BULL_SPS, 0.0
    else:
        n_secs = max(3 - n_compete, 1)
        sps, cash = BEAR_SPS, 0.20

    top_secs = sec_mom.head(n_secs).index.tolist()
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    stock_frac = max(1.0 - cash - total_compete, 0.0)
    if not selected:
        w = {}
        if gld_a > 0: w['GLD'] = gld_a
        if gdx_a > 0: w['GDX'] = gdx_a
        return w

    iv   = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw   = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}
    weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a > 0: weights['GLD'] = gld_a
    if gdx_a > 0: weights['GDX'] = gdx_a
    return weights


def apply_overlays_with_vol(weights, spy_vol, dd, port_vol_ann):
    """Apply GDXJ vol-trigger, GLD DD response, and Portfolio Vol Targeting"""
    # GDXJ vol trigger
    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0

    # GLD DD response
    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)

    total = gdxj_v + gld_dd
    if total > 0 and weights:
        stock_frac = max(1.0 - total, 0.01)
        tot = sum(weights.values())
        if tot > 0:
            weights = {t: w/tot*stock_frac for t, w in weights.items()}
        if gld_dd > 0: weights['GLD'] = weights.get('GLD', 0) + gld_dd
        if gdxj_v > 0: weights['GDXJ'] = weights.get('GDXJ', 0) + gdxj_v

    # Portfolio Volatility Targeting
    shy_boost = 0.0
    if port_vol_ann > 0.01:
        scale = min(VOL_TARGET_ANN / max(port_vol_ann, 0.01), 1.0)
        if scale < 0.98:
            equity_keys = [t for t in weights if t not in ('GLD', 'GDX', 'GDXJ')]
            eq_frac = sum(weights[t] for t in equity_keys)
            if eq_frac > 0:
                for t in equity_keys:
                    weights[t] = weights[t] * scale
                shy_boost = eq_frac * (1.0 - scale)

    return weights, shy_boost


def run_daily_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                       start='2015-01-01', end='2025-12-31', cost=0.0015):
    """月频调仓 + 日频净值追踪（vol targeting 每月末更新）"""

    all_daily = close_df.loc[start:end].dropna(how='all')
    month_ends = all_daily.resample('ME').last().index
    trading_days = all_daily.index
    SPY_VOL = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None

    val = 1.0; peak = 1.0
    equity_vals, equity_dates = [], []
    current_weights = {}   # includes shy_boost as 'SHY'
    prev_hold, prev_w = set(), {}
    processed_months = set()
    port_returns = []       # monthly returns for vol targeting
    scale_hist = []

    for day_idx, day in enumerate(trading_days):
        # ── Check if we should rebalance today (first trading day after a month-end) ──
        past_month_ends = month_ends[month_ends < day]
        if len(past_month_ends) > 0:
            last_me = past_month_ends[-1]
            next_days_after_me = trading_days[trading_days > last_me]
            execution_day = next_days_after_me[0] if len(next_days_after_me) > 0 else None

            if execution_day is not None and day == execution_day and last_me not in processed_months:
                # ── Rebalance: signal computed at last_me ──
                dd_now = (val - peak) / peak if peak > 0 else 0
                spy_vol = float(SPY_VOL.loc[:last_me].dropna().iloc[-1]) if (
                    SPY_VOL is not None and len(SPY_VOL.loc[:last_me].dropna()) > 0) else 0.15

                # Vol targeting: use past VOL_LOOKBACK months of portfolio returns
                if len(port_returns) >= VOL_LOOKBACK:
                    port_vol_mon = np.std(port_returns[-VOL_LOOKBACK:], ddof=1)
                    port_vol_ann = port_vol_mon * np.sqrt(12)
                    scale_val = min(VOL_TARGET_ANN / max(port_vol_ann, 0.01), 1.0)
                    scale_hist.append(scale_val)
                else:
                    port_vol_ann = 0.20  # bootstrap assumption

                new_w = select(sig, sectors, last_me, prev_hold, gld_p, gdx_p)
                new_w, shy_boost = apply_overlays_with_vol(new_w, spy_vol, dd_now, port_vol_ann)
                if shy_boost > 0:
                    new_w['SHY'] = new_w.get('SHY', 0) + shy_boost

                # Turnover cost
                all_t = set(new_w) | set(prev_w)
                turnover = sum(abs(new_w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
                val *= (1 - turnover * cost * 2)

                current_weights = new_w.copy()
                prev_w = new_w.copy()
                prev_hold = {k for k in new_w if k not in ('GLD', 'GDX', 'GDXJ', 'SHY')}
                processed_months.add(last_me)

                # Record month return: from last processed rebalance to now
                # We'll track it by appending once per month at end of holding period

        # ── Daily P&L ──
        if day_idx == 0:
            equity_vals.append(val)
            equity_dates.append(day)
            continue

        prev_day = trading_days[day_idx - 1]
        day_ret = 0.0; invested = 0.0

        for ticker, w in current_weights.items():
            if   ticker == 'GLD':  series = gld_p
            elif ticker == 'GDX':  series = gdx_p
            elif ticker == 'GDXJ': series = gdxj_p
            elif ticker == 'SHY':  series = shy_p
            elif ticker in close_df.columns: series = close_df[ticker]
            else: continue

            if (prev_day in series.index and day in series.index):
                p0 = series.loc[prev_day]; p1 = series.loc[day]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    day_ret += (p1 / p0 - 1) * w
                    invested += w

        # Residual cash → SHY
        cash_frac = max(1.0 - invested, 0.0)
        if cash_frac > 0.01 and shy_p is not None and USE_SHY:
            if prev_day in shy_p.index and day in shy_p.index:
                sp, st = shy_p.loc[prev_day], shy_p.loc[day]
                if pd.notna(sp) and pd.notna(st) and sp > 0:
                    day_ret += (st / sp - 1) * cash_frac

        val *= (1 + day_ret)
        peak = max(peak, val)
        equity_vals.append(val)
        equity_dates.append(day)

        # Track monthly returns for vol targeting: record at month-end
        if day in month_ends:
            # Find the month start (previous month-end or start date)
            prev_me_list = month_ends[month_ends < day]
            if len(prev_me_list) > 0:
                prev_me = prev_me_list[-1]
                # month return: from day after prev_me to day
                prev_me_pos = trading_days.get_loc(prev_me) if prev_me in trading_days else None
                if prev_me_pos is not None:
                    val_at_prev_me = equity_vals[prev_me_pos]
                    mon_ret = val / val_at_prev_me - 1
                    port_returns.append(mon_ret)

    eq = pd.Series(equity_vals, index=pd.DatetimeIndex(equity_dates))
    avg_scale = float(np.mean(scale_hist)) if scale_hist else 1.0
    pct_scaled = float(np.mean([s < 0.99 for s in scale_hist])) if scale_hist else 0.0
    return eq, avg_scale, pct_scaled


def compute_metrics(eq, rf=0.04):
    if len(eq) < 30: return {}
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs < 0.5: return {}
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1
    running_max = eq.cummax()
    drawdown = (eq - running_max) / running_max
    max_dd = drawdown.min()
    daily_rets = eq.pct_change().dropna()
    ann_ret = daily_rets.mean() * 252
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    composite = sharpe * 0.4 + calmar * 0.4 + cagr * 0.2
    dd_end_idx = drawdown.idxmin()
    dd_start_idx = eq.loc[:dd_end_idx].idxmax()
    return dict(cagr=float(cagr), max_dd=float(max_dd), sharpe=float(sharpe),
                calmar=float(calmar), composite=float(composite),
                max_dd_start=str(dd_start_idx.date()), max_dd_end=str(dd_end_idx.date()),
                ann_vol=float(ann_vol), final_val=float(eq.iloc[-1]))


def main():
    print("=" * 72)
    print("🐻 动量轮动 v9i-Daily — 日频净值追踪版本（波动率目标化）")
    print("=" * 72)

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    sig = precompute(close_df)
    print(f"  加载 {len(close_df.columns)} 支股票")

    # ── 全期（2015-2025）日频回测 ──
    print("\n🔄 日频全期回测 (2015-2025)...")
    eq_full, avg_sc, pct_sc = run_daily_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
        start='2015-01-01', end='2025-12-31')
    m_full = compute_metrics(eq_full)

    # ── IS/OOS 分割 ──
    print("🔄 IS 回测 (2015-2020)...")
    eq_is, _, _ = run_daily_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
        start='2015-01-01', end='2020-12-31')
    m_is = compute_metrics(eq_is)

    print("🔄 OOS 回测 (2021-2025)...")
    eq_oos, _, _ = run_daily_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
        start='2021-01-01', end='2025-12-31')
    m_oos = compute_metrics(eq_oos)

    wf = m_oos['sharpe'] / m_is['sharpe'] if m_is.get('sharpe', 0) > 0 else 0

    # ── 参数敏感性测试 ──
    print("\n🔄 参数敏感性：target_vol ± 2%...")
    results_sens = {}
    for tv in [0.09, 0.11, 0.13]:
        global VOL_TARGET_ANN
        VOL_TARGET_ANN = tv
        eq_s, _, _ = run_daily_backtest(
            close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p)
        results_sens[f"tv={tv:.0%}"] = compute_metrics(eq_s)
        print(f"  target_vol={tv:.0%}: CAGR={results_sens[f'tv={tv:.0%}']['cagr']:.1%}, "
              f"MaxDD={results_sens[f'tv={tv:.0%}']['max_dd']:.1%}, "
              f"Composite={results_sens[f'tv={tv:.0%}']['composite']:.3f}")
    VOL_TARGET_ANN = 0.11  # reset

    print("\n" + "=" * 72)
    print("📊 v9i-Daily 结果（全期 2015-2025）")
    print("=" * 72)
    print(f"  CAGR:      {m_full['cagr']:.2%}")
    print(f"  MaxDD:     {m_full['max_dd']:.2%}  ← 真实日频值")
    print(f"  Sharpe:    {m_full['sharpe']:.2f}")
    print(f"  Calmar:    {m_full['calmar']:.2f}")
    print(f"  Composite: {m_full['composite']:.4f}")
    print(f"  MaxDD区间: {m_full.get('max_dd_start','')} → {m_full.get('max_dd_end','')}")
    print(f"  avg_scale: {avg_sc:.3f}")
    print(f"  pct_scaled:{pct_sc:.0%}")
    print(f"\n  IS Sharpe:  {m_is.get('sharpe', 0):.2f}")
    print(f"  OOS Sharpe: {m_oos.get('sharpe', 0):.2f}")
    print(f"  WF ratio:   {wf:.2f}")

    print("\n📊 对比表")
    print("  " + "─" * 60)
    print(f"  {'指标':<12} {'月频(v9i)':>12} {'日频(v9i)':>12} {'日频(v9g)':>12}")
    print("  " + "─" * 60)
    print(f"  {'CAGR':<12} {'31.9%':>12} {m_full['cagr']:>11.1%} {'35.39%':>12}")
    print(f"  {'MaxDD':<12} {'-10.7%':>12} {m_full['max_dd']:>11.1%} {'-26.51%':>12}")
    print(f"  {'Sharpe':<12} {'1.81':>12} {m_full['sharpe']:>12.2f} {'1.35':>12}")
    print(f"  {'Composite':<12} {'1.973':>12} {m_full['composite']:>12.3f} {'1.146':>12}")
    print("  " + "─" * 60)

    out = {
        'strategy': 'v9i-Daily (月频调仓+日频净值追踪)',
        'full': m_full, 'is': m_is, 'oos': m_oos,
        'wf': float(wf),
        'vol_targeting': {'avg_scale': float(avg_sc), 'pct_scaled': float(pct_sc)},
        'sensitivity': results_sens,
    }
    jf = Path(__file__).parent / "momentum_v9i_daily_results.json"
    jf.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n💾 Results → {jf}")
    return out


if __name__ == '__main__':
    main()
