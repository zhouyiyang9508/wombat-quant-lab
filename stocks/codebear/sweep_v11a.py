#!/usr/bin/env python3
"""
v11a 联合参数优化扫描
代码熊 🐻 | 2026-02-22

目标：在 v11a Master 基础上，联合扫描三大创新层的核心参数：
  ① DEFENSIVE_FRAC    — 软牛期防御 ETF 分配（XLV/XLP/XLU）
  ② SPY_SOFT_HI_FRAC  — SPY 月跌>7% 时 GLD 软对冲分配
  ③ TLT/IEF 熊市分配  — 保持 v10d 最优值（25%/20%），作为固定基准

扫描范围：
  DEFENSIVE_FRAC : [0.10, 0.12, 0.15, 0.18, 0.20]
  SPY_SOFT_HI   : [0.06, 0.08, 0.10, 0.12, 0.15]

共 5×5 = 25 个配置，与 v11a 冠军 (0.15, 0.10) 对比

筛选标准：
  - Composite > 2.10（不低于 v10d）
  - WF > 0.70（不低于 v11a 当前值）
"""

import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util
import itertools

warnings.filterwarnings('ignore')

BASE        = Path(__file__).resolve().parent.parent.parent
CACHE       = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"


def load_csv(fp):
    df = pd.read_csv(fp)
    c  = 'Date' if 'Date' in df.columns else df.columns[0]
    df[c] = pd.to_datetime(df[c])
    return df.set_index(c).sort_index()


def load_stocks(tickers):
    d = {}
    for t in tickers:
        f = STOCK_CACHE / f"{t}.csv"
        if not f.exists() or f.stat().st_size < 500: continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200:
                d[t] = pd.to_numeric(df['Close'], errors='coerce').dropna()
        except: pass
    return pd.DataFrame(d)


def calc_composite(m):
    # get_metrics 返回的是 max_dd（负值），calmar = cagr / |max_dd|
    return m['sharpe'] * 0.4 + m['calmar'] * 0.4 + min(m['cagr'], 1.0) * 0.2




def load_v11a():
    spec = importlib.util.spec_from_file_location(
        "v11a", Path(__file__).parent / "momentum_v11a_master.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_with_params(v11a, close_df, sig, sectors,
                    gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
                    def_frac, spy_soft_frac):
    """Monkey-patch v11a globals，运行 full + IS + OOS 三次回测"""
    v11a.DEFENSIVE_FRAC = def_frac
    v11a.DEFENSIVE_EACH = def_frac / len(v11a.DEFENSIVE_ETFS)
    v11a.SPY_SOFT_HI_FRAC = spy_soft_frac

    def _run(s, e):
        eq, _, _, _ = v11a.run_backtest(
            close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
            shy_p, tlt_p, ief_p, def_prices, s, e)
        return eq, v11a.compute_metrics(eq)

    eq_full, m_full = _run('2015-01-01', '2025-12-31')
    eq_is,   m_is   = _run('2015-01-01', '2020-12-31')
    eq_oos,  m_oos  = _run('2021-01-01', '2025-12-31')

    wf = round(m_oos['sharpe'] / m_is['sharpe'], 3) if m_is['sharpe'] > 0 else 0
    return m_full, m_is['sharpe'], m_oos['sharpe'], wf


def main():
    print("=" * 80)
    print("v11a 联合参数优化扫描 🐻")
    print("=" * 80)

    # ── 加载数据（只需加载一次）──────────────────────────────────────────────
    print("\n[1/3] 加载数据...")
    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p    = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p    = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p   = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p    = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    tlt_p    = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    ief_p    = load_csv(CACHE / "IEF.csv")['Close'].dropna()
    xlv_p    = load_csv(CACHE / "XLV.csv")['Close'].dropna()
    xlp_p    = load_csv(CACHE / "XLP.csv")['Close'].dropna()
    xlu_p    = load_csv(CACHE / "XLU.csv")['Close'].dropna()
    def_prices = {'XLV': xlv_p, 'XLP': xlp_p, 'XLU': xlu_p}
    print(f"  ✓ {len(close_df.columns)} 只股票，{len(close_df)} 个交易日")

    v11a = load_v11a()
    sig  = v11a.precompute(close_df)
    print("  ✓ 信号预计算完成")

    # ── 参数网格 ──────────────────────────────────────────────────────────────
    def_fracs      = [0.10, 0.12, 0.15, 0.18, 0.20]
    spy_soft_fracs = [0.06, 0.08, 0.10, 0.12, 0.15]
    combos = list(itertools.product(def_fracs, spy_soft_fracs))
    print(f"\n[2/3] 参数扫描：{len(combos)} 个配置...")
    print(f"  DEFENSIVE_FRAC   : {def_fracs}")
    print(f"  SPY_SOFT_HI_FRAC : {spy_soft_fracs}")
    print(f"  (★ = v11a 当前冠军配置 0.15/0.10)")

    results = []
    champion = {'Composite': 2.160, 'WF': 0.74}  # v11a 当前冠军

    for i, (df_frac, ss_frac) in enumerate(combos):
        tag = " ★" if (df_frac == 0.15 and ss_frac == 0.10) else ""
        print(f"  [{i+1:02d}/{len(combos)}] def={df_frac:.0%} spy_soft={ss_frac:.0%}{tag}", end=' ', flush=True)

        try:
            m, is_s, oos_s, wf = run_with_params(
                v11a, close_df, sig, sectors,
                gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
                df_frac, ss_frac
            )
            composite = calc_composite(m)

            row = {
                'def_frac':   df_frac,
                'spy_soft':   ss_frac,
                'cagr':       round(m['cagr'], 4),
                'maxdd':      round(m['max_dd'], 4),
                'sharpe':     round(m['sharpe'], 3),
                'calmar':     round(m['calmar'], 3),
                'composite':  round(composite, 4),
                'is_sharpe':  round(is_s, 3),
                'oos_sharpe': round(oos_s, 3),
                'wf':         round(wf, 3),
            }
            results.append(row)
            beat = " 🏆" if composite > champion['Composite'] and wf >= 0.70 else ""
            print(f"→ Composite={composite:.3f}  WF={wf:.2f}{beat}")
            if composite > champion['Composite'] and wf >= 0.70:
                champion = {'Composite': composite, 'WF': wf, **row}
        except Exception as e:
            print(f"→ ERROR: {e}")

    # ── 结果汇总 ──────────────────────────────────────────────────────────────
    print(f"\n[3/3] 结果汇总")
    results.sort(key=lambda x: x['composite'], reverse=True)

    print("\n" + "=" * 100)
    print(f"{'def%':>5} {'soft%':>6} {'CAGR':>7} {'MaxDD':>7} "
          f"{'Sharpe':>7} {'Calmar':>7} {'Composite':>10} {'WF':>6} "
          f"{'IS':>6} {'OOS':>6}  备注")
    print("-" * 100)

    for r in results[:20]:  # 显示前 20
        wf_ok  = "✅" if r['wf'] >= 0.70 else ("⚠️" if r['wf'] >= 0.60 else "❌")
        is_champ = (r['def_frac'] == 0.15 and r['spy_soft'] == 0.10)
        flag = " ★ v11a基准" if is_champ else ""
        if r['composite'] > 2.160 and r['wf'] >= 0.70:
            flag += " 🏆 新冠军"
        print(f"{r['def_frac']:>4.0%}  {r['spy_soft']:>5.0%}  "
              f"{r['cagr']:>6.1%}  {r['maxdd']:>6.1%}  "
              f"{r['sharpe']:>6.3f}  {r['calmar']:>6.3f}  "
              f"{r['composite']:>9.3f}  {wf_ok}{r['wf']:>4.2f}  "
              f"{r['is_sharpe']:>5.3f}  {r['oos_sharpe']:>5.3f} {flag}")

    # 最优结果
    valid = [r for r in results if r['wf'] >= 0.70]
    best  = max(valid, key=lambda x: x['composite']) if valid else results[0]
    base  = next((r for r in results if r['def_frac'] == 0.15 and r['spy_soft'] == 0.10), None)

    print("\n" + "=" * 100)
    print(f"\n🏆 最优配置（WF ≥ 0.70）：def={best['def_frac']:.0%}，spy_soft={best['spy_soft']:.0%}")
    print(f"   Composite={best['composite']:.3f}  Sharpe={best['sharpe']:.3f}  "
          f"CAGR={best['cagr']:.1%}  MaxDD={best['maxdd']:.1%}  WF={best['wf']:.2f}")
    if base:
        delta = best['composite'] - base['composite']
        print(f"\n对比 v11a 基准（def=15%，spy_soft=10%）：")
        print(f"   Composite: {base['composite']:.3f} → {best['composite']:.3f} "
              f"({'+'if delta>=0 else ''}{delta:.3f})")
        print(f"   WF:        {base['wf']:.2f} → {best['wf']:.2f}")

    # 保存
    out_path = Path(__file__).parent / "sweep_v11a_results.json"
    with open(out_path, 'w') as f:
        json.dump({
            'sweep': results,
            'best':  best,
            'baseline': base,
            'meta': {
                'date': '2026-02-22',
                'author': '代码熊 🐻',
                'params': {
                    'def_fracs': def_fracs,
                    'spy_soft_fracs': spy_soft_fracs,
                    'tlt_bear_frac': 0.25,
                    'ief_bear_frac': 0.20,
                }
            }
        }, f, indent=2)
    print(f"\n📁 结果已保存：{out_path}")
    return results, best


if __name__ == '__main__':
    results, best = main()
