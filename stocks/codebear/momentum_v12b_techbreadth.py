"""
Momentum v12b — 科技宽度实验
================================
基于 v11b_final，测试两种科技宽度方案：
  方案A: 用科技板块宽度替代全市场宽度（79只科技股 > 50dMA 的比例）
  方案B: 双轨豁免（科技宽度>70% OR 整体宽度>65% → bull_hi）

背景:
  - 整体宽度在 2023-2024 AI 牛市中可能只有 50-60%（工业/能源疲软）
  - 动量策略持有的恰好是高动量科技股，科技宽度此时 >70%
  - 整体宽度 < 0.65 → 触发 soft_bull → 加入 XLV/XLP/XLU 防御仓位（错误的）
  - 方案A/B 旨在减少此类误触发，保持满仓高动量科技

代码熊 🐻 | 2026-02-22
"""

import pandas as pd, numpy as np, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import stocks.codebear.momentum_v11b_final as v11b

CACHE = Path(__file__).parent.parent.parent / 'data_cache'

BREADTH_NARROW = v11b.BREADTH_NARROW   # 0.45
BREADTH_CONC   = v11b.BREADTH_CONC     # 0.65
TECH_BULL_HI   = 0.70                  # 科技宽度豁免阈值


def get_tech_tickers(sectors):
    return [t for t, s in sectors.items() if s == 'Technology']


def compute_tech_breadth(sig, date, tech_tickers):
    """科技股中 > 50dMA 的比例"""
    close = sig['close'].loc[:date].dropna(how='all')
    sma50 = sig['sma50'].loc[:date].dropna(how='all')
    if len(close) < 50: return 1.0
    lc = close.iloc[-1]; ls = sma50.iloc[-1]
    valid = [t for t in tech_tickers if t in lc.index and t in ls.index]
    if not valid: return 1.0
    mask = pd.Series({t: lc[t] > ls[t] for t in valid}).dropna()
    return float(mask.sum() / len(mask)) if len(mask) > 0 else 1.0


def make_regime_A(tech_tickers):
    """方案A: 科技宽度替代整体宽度"""
    def get_regime(sig, date):
        if sig['s200'] is None: return 'bull_hi', 1.0
        spy_now  = sig['spy'].loc[:date].dropna()
        s200_now = sig['s200'].loc[:date].dropna()
        if len(spy_now) == 0: return 'bull_hi', 1.0
        tech_b = compute_tech_breadth(sig, date, tech_tickers)
        if spy_now.iloc[-1] < s200_now.iloc[-1] and tech_b < BREADTH_NARROW:
            return 'bear', tech_b
        elif tech_b < BREADTH_CONC:
            return 'soft_bull', tech_b
        else:
            return 'bull_hi', tech_b
    return get_regime


def make_regime_B(tech_tickers):
    """方案B: 双轨豁免 — 科技>70% 或 整体>65% 都算 bull_hi"""
    def get_regime(sig, date):
        if sig['s200'] is None: return 'bull_hi', 1.0
        spy_now  = sig['spy'].loc[:date].dropna()
        s200_now = sig['s200'].loc[:date].dropna()
        if len(spy_now) == 0: return 'bull_hi', 1.0
        overall_b = v11b.compute_breadth(sig, date)
        tech_b    = compute_tech_breadth(sig, date, tech_tickers)
        # 熊市需三条件同时满足
        if (spy_now.iloc[-1] < s200_now.iloc[-1]
                and overall_b < BREADTH_NARROW
                and tech_b < BREADTH_NARROW):
            return 'bear', overall_b
        # 双轨豁免: 任一宽度达标 → bull_hi
        if overall_b >= BREADTH_CONC or tech_b >= TECH_BULL_HI:
            return 'bull_hi', overall_b
        return 'soft_bull', overall_b
    return get_regime


def run_with_regime(regime_fn, close_df, sig, sectors,
                    gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
                    start='2015-01-01', end='2025-12-31'):
    """Monkeypatch regime 函数后运行 v11b 回测"""
    orig = v11b.get_three_regime
    v11b.get_three_regime = regime_fn
    try:
        eq, avg_to, regime_hist, bond_hist = v11b.run_backtest(
            close_df, sig, sectors,
            gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
            start=start, end=end
        )
    finally:
        v11b.get_three_regime = orig
    return eq, avg_to, regime_hist


def compute_wf(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
               tlt_p, ief_p, def_prices, regime_fn=None):
    """计算 IS/OOS Walk-Forward（IS=2015-2021, OOS=2022-2025）"""
    orig = None
    if regime_fn is not None:
        orig = v11b.get_three_regime
        v11b.get_three_regime = regime_fn

    try:
        eq_is, _, _, _ = v11b.run_backtest(
            close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
            tlt_p, ief_p, def_prices, '2015-01-01', '2021-12-31')
        eq_oos, _, _, _ = v11b.run_backtest(
            close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
            tlt_p, ief_p, def_prices, '2022-01-01', '2025-12-31')
    finally:
        if orig is not None:
            v11b.get_three_regime = orig

    m_is  = v11b.compute_metrics(eq_is)
    m_oos = v11b.compute_metrics(eq_oos)
    wf = m_oos['sharpe'] / m_is['sharpe'] if m_is['sharpe'] > 0 else 0.0
    return wf, m_is, m_oos


def main():
    print("=" * 72)
    print("🐻 v12b — 科技宽度实验 (方案A vs 方案B vs Baseline v11b)")
    print("=" * 72)

    # 加载数据
    print("\n[1/3] 加载数据...")
    with open(CACHE / 'sp500_tickers.txt') as f:
        tickers = [t.strip() for t in f if t.strip()]
    with open(CACHE / 'sp500_sectors.json') as f:
        sectors = json.load(f)

    tech_tickers = get_tech_tickers(sectors)
    print(f"  科技股: {len(tech_tickers)} 只 | 总股票池: {len(tickers)} 只")

    close_df = v11b.load_stocks(tickers + ['SPY'])
    stock_cols = [c for c in close_df.columns if c not in ('GLD','GDX','GDXJ','SHY','TLT','IEF')]

    def lc(name):
        fp = CACHE / f'{name}.csv'
        if not fp.exists(): return pd.Series(dtype=float)
        df = pd.read_csv(fp); c = 'Date' if 'Date' in df.columns else df.columns[0]
        df[c] = pd.to_datetime(df[c])
        return df.set_index(c).sort_index()['Close'].dropna()

    gld_p  = lc('GLD'); gdx_p  = lc('GDX'); gdxj_p = lc('GDXJ')
    shy_p  = lc('SHY'); tlt_p  = lc('TLT'); ief_p  = lc('IEF')
    def_prices = {'XLV': lc('XLV'), 'XLP': lc('XLP'), 'XLU': lc('XLU')}

    print("[2/3] 预计算信号...")
    sig = v11b.precompute(close_df[stock_cols])

    print("[3/3] 运行三种策略 + WF 计算...\n")

    configs = [
        ('Baseline v11b',  None),
        ('方案A 科技宽度', make_regime_A(tech_tickers)),
        ('方案B 双轨豁免', make_regime_B(tech_tickers)),
    ]

    rows = []
    for name, regime_fn in configs:
        print(f"  [{name}] 全期回测...")
        if regime_fn is None:
            eq, _, rh = run_with_regime(v11b.get_three_regime, close_df[stock_cols], sig, sectors,
                                         gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices)
        else:
            eq, _, rh = run_with_regime(regime_fn, close_df[stock_cols], sig, sectors,
                                         gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices)
        m = v11b.compute_metrics(eq)

        print(f"  [{name}] WF 计算...")
        wf, m_is, m_oos = compute_wf(close_df[stock_cols], sig, sectors,
                                      gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
                                      regime_fn=regime_fn)
        total_r = sum(rh.values())
        rows.append({
            'name': name,
            'm': m, 'wf': wf,
            'is_sh': m_is['sharpe'], 'oos_sh': m_oos['sharpe'],
            'is_cagr': m_is['cagr'], 'oos_cagr': m_oos['cagr'],
            'bull_hi_pct': rh.get('bull_hi', 0) / total_r if total_r else 0,
            'soft_bull_pct': rh.get('soft_bull', 0) / total_r if total_r else 0,
            'bear_pct': rh.get('bear', 0) / total_r if total_r else 0,
        })

    # ── 输出 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("📊 结果汇总 (月频，IS=2015-2021，OOS=2022-2025)")
    print("=" * 80)
    print(f"{'策略':>16} | {'CAGR':>6} {'MaxDD':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'WF':>6} {'IS_Sh':>6} {'OOS_Sh':>7}")
    print("-" * 80)
    for r in rows:
        m = r['m']
        flag = "✅" if r['wf'] >= 0.70 else ("⚠️" if r['wf'] >= 0.60 else "❌")
        print(f"  {r['name']:>14} | {m['cagr']:>5.1%} {m['max_dd']:>7.1%} "
              f"{m['sharpe']:>7.2f} {m['calmar']:>7.2f} "
              f"{r['wf']:>5.3f}{flag} {r['is_sh']:>6.2f} {r['oos_sh']:>7.2f}")

    print("\n📐 Regime 触发频率:")
    print(f"{'策略':>16} | {'bull_hi':>8} {'soft_bull':>10} {'bear':>6}")
    print("-" * 48)
    for r in rows:
        print(f"  {r['name']:>14} | {r['bull_hi_pct']:>7.1%} {r['soft_bull_pct']:>10.1%} {r['bear_pct']:>6.1%}")

    print("\n📈 IS vs OOS CAGR:")
    print(f"{'策略':>16} | {'IS_CAGR(15-21)':>15} {'OOS_CAGR(22-25)':>16}")
    print("-" * 52)
    for r in rows:
        print(f"  {r['name']:>14} | {r['is_cagr']:>15.1%} {r['oos_cagr']:>16.1%}")

    print("\n✅ 结论:")
    base = rows[0]
    for r in rows[1:]:
        wf_delta = r['wf'] - base['wf']
        cagr_delta = r['m']['cagr'] - base['m']['cagr']
        dd_delta = r['m']['max_dd'] - base['m']['max_dd']
        print(f"  {r['name']}: WF {'+' if wf_delta>=0 else ''}{wf_delta:.3f} | "
              f"CAGR {'+' if cagr_delta>=0 else ''}{cagr_delta:.1%} | "
              f"MaxDD {'+' if dd_delta>=0 else ''}{dd_delta:.1%}")


if __name__ == '__main__':
    main()
