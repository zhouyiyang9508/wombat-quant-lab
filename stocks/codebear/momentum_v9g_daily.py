#!/usr/bin/env python3
"""
动量轮动 v9g-Daily — 日频净值追踪版本
代码熊 🐻

基于 v9g_final 的选股逻辑（动态行业集中度），改为月频调仓 + 日频净值追踪
修复原始月频回测中 MaxDD 严重低估的缺陷

📊 性能对比 (2015-2025):
               月频(原始)    日频(真实)
  CAGR:        37.18%       37.03%
  MaxDD:       -14.88%      -26.51% ← 真实值！低估1.78倍
  Sharpe:      1.71         1.37
  Calmar:      2.50         1.40
  Composite:   1.7589       1.1812

最大回撤区间: 2020-02-20 → 2020-03-20 (COVID-19)

关键发现：
  月频回测只看月末快照，完全看不到2020年3月的月中暴跌
  日频追踪揭示真实MaxDD = -26.51%，是月频报告的1.78倍
  Calmar 从 2.50 降至 1.40（分母被低估导致Calmar虚高）

严格约束：
  - 今日净值 = close[i-1] → close[i] 的涨跌
  - 月末调仓信号基于当月末收盘价，T+1执行
  - 时间段 2015-01 ~ 2025-12
"""

import json, warnings, sys
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"


def load_strategy_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def run_daily_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                       select_fn, apply_overlays_fn, get_spy_vol_fn,
                       start='2015-01-01', end='2025-12-31',
                       cost=0.0015, stop_loss=None):
    """月频调仓 + 日频净值追踪"""
    all_daily = close_df.loc[start:end].dropna(how='all')
    month_ends = all_daily.resample('ME').last().index
    trading_days = all_daily.index

    val = 1.0; peak = 1.0
    equity_vals, equity_dates = [], []
    current_weights = {}
    prev_hold, prev_w = set(), {}
    month_start_val = 1.0
    processed_months = set()
    stop_loss_active = False

    for day_idx, day in enumerate(trading_days):
        past_month_ends = month_ends[month_ends < day]
        if len(past_month_ends) > 0:
            last_me = past_month_ends[-1]
            next_days_after_me = trading_days[trading_days > last_me]
            if len(next_days_after_me) > 0:
                execution_day = next_days_after_me[0]
            else:
                execution_day = None

            if execution_day is not None and day == execution_day and last_me not in processed_months:
                dd_for_signal = (val - peak) / peak if peak > 0 else 0
                spy_vol = get_spy_vol_fn(sig, last_me)
                new_weights = select_fn(sig, sectors, last_me, prev_hold, gld_p, gdx_p)
                new_weights = apply_overlays_fn(new_weights, spy_vol, dd_for_signal)

                all_t = set(new_weights) | set(prev_w)
                turnover = sum(abs(new_weights.get(t,0) - prev_w.get(t,0)) for t in all_t) / 2
                val *= (1 - turnover * cost * 2)

                current_weights = new_weights.copy()
                prev_w = new_weights.copy()
                prev_hold = {k for k in new_weights if k not in ('GLD','GDX','GDXJ','SHY')}
                month_start_val = val
                stop_loss_active = False
                processed_months.add(last_me)

        if day_idx == 0:
            equity_vals.append(val)
            equity_dates.append(day)
            continue

        prev_day = trading_days[day_idx - 1]
        day_ret = 0.0; invested = 0.0

        for ticker, w in current_weights.items():
            if ticker == 'GLD': series = gld_p
            elif ticker == 'GDX': series = gdx_p
            elif ticker == 'GDXJ': series = gdxj_p
            elif ticker == 'SHY': series = shy_p
            elif ticker in close_df.columns: series = close_df[ticker]
            else: continue

            if prev_day in series.index and day in series.index:
                p_prev = series.loc[prev_day]
                p_today = series.loc[day]
                if pd.notna(p_prev) and pd.notna(p_today) and p_prev > 0:
                    day_ret += (p_today / p_prev - 1) * w
                    invested += w

        cash_frac = max(1.0 - invested, 0.0)
        if cash_frac > 0.01 and shy_p is not None:
            if prev_day in shy_p.index and day in shy_p.index:
                sp, st = shy_p.loc[prev_day], shy_p.loc[day]
                if pd.notna(sp) and pd.notna(st) and sp > 0:
                    day_ret += (st / sp - 1) * cash_frac

        val *= (1 + day_ret)
        peak = max(peak, val)

        if stop_loss is not None and not stop_loss_active and month_start_val > 0:
            month_ret = val / month_start_val - 1
            if month_ret < stop_loss:
                current_weights = {'SHY': 1.0}
                stop_loss_active = True

        equity_vals.append(val)
        equity_dates.append(day)

    return pd.Series(equity_vals, index=pd.DatetimeIndex(equity_dates))


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
    print("=" * 70)
    print("🐻 动量轮动 v9g-Daily — 日频净值追踪版本")
    print("=" * 70)

    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    print(f"  加载 {len(close_df.columns)} 支股票")

    v9g_mod = load_strategy_module("v9g", Path(__file__).parent / "momentum_v9g_final.py")
    sig = v9g_mod.precompute(close_df)

    # A. 无止损
    print("\n🔄 日频回测 (无止损)...")
    eq_A = run_daily_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                              v9g_mod.select, v9g_mod.apply_overlays, v9g_mod.get_spy_vol)
    m_A = compute_metrics(eq_A)

    # B. 止损 -15%
    print("🔄 日频回测 (止损 -15%)...")
    eq_B = run_daily_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                              v9g_mod.select, v9g_mod.apply_overlays, v9g_mod.get_spy_vol,
                              stop_loss=-0.15)
    m_B = compute_metrics(eq_B)

    # C. 止损 -10%
    print("🔄 日频回测 (止损 -10%)...")
    eq_C = run_daily_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                              v9g_mod.select, v9g_mod.apply_overlays, v9g_mod.get_spy_vol,
                              stop_loss=-0.10)
    m_C = compute_metrics(eq_C)

    # IS/OOS splits
    print("🔄 IS/OOS 分割...")
    eq_IS = run_daily_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                               v9g_mod.select, v9g_mod.apply_overlays, v9g_mod.get_spy_vol,
                               start='2015-01-01', end='2020-12-31')
    eq_OOS = run_daily_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                v9g_mod.select, v9g_mod.apply_overlays, v9g_mod.get_spy_vol,
                                start='2021-01-01', end='2025-12-31')
    m_IS = compute_metrics(eq_IS)
    m_OOS = compute_metrics(eq_OOS)
    wf = m_OOS['sharpe'] / m_IS['sharpe'] if m_IS.get('sharpe', 0) > 0 else 0

    print("\n" + "=" * 70)
    print("📊 v9g-Daily 结果")
    print("=" * 70)
    for label, m in [("无止损", m_A), ("止损-15%", m_B), ("止损-10%", m_C)]:
        print(f"\n  [{label}]")
        print(f"    CAGR:      {m['cagr']:.2%}")
        print(f"    MaxDD:     {m['max_dd']:.2%}")
        print(f"    Sharpe:    {m['sharpe']:.2f}")
        print(f"    Calmar:    {m['calmar']:.2f}")
        print(f"    Composite: {m['composite']:.4f}")
        if 'max_dd_start' in m:
            print(f"    DD区间:    {m['max_dd_start']} → {m['max_dd_end']}")

    print(f"\n  IS Sharpe:  {m_IS.get('sharpe', 0):.2f}")
    print(f"  OOS Sharpe: {m_OOS.get('sharpe', 0):.2f}")
    print(f"  WF ratio:   {wf:.2f}")

    out = {
        'strategy': 'v9g-Daily (日频净值追踪)',
        'no_stop': m_A, 'sl15': m_B, 'sl10': m_C,
        'is': m_IS, 'oos': m_OOS, 'wf': float(wf),
    }
    jf = Path(__file__).parent / "momentum_v9g_daily_results.json"
    jf.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n💾 Results → {jf}")
    return out


if __name__ == '__main__':
    main()
