"""
Hybrid v6 — v11b 股票端 + v5-D Crypto（阈值切换+独立MA）
==========================================================
目标：利用 v11b 更强的防御能力（MaxDD -20.8%）提升 Hybrid 整体 MaxDD，
      同时保留 v5-D 在 WF 上的优势（0.640 vs v3 的 0.559）。

预期：
  - MaxDD 从 -22.1%（v5-D + v9j）改善到约 -19~20%
  - WF 维持 ≥ 0.60
  - CAGR 与 v5-D + v9j 持平或略低（v11b 月频 ~32% vs v9j ~32%，相近）

数据来源：
  - StockV11b: hybrid_v4_equity.csv（已含日频数据）
  - Crypto v5-D: hybrid_v5_dynbtceth.py（已修复前瞻偏差）

代码熊 🐻 | 2026-02-22
"""

import pandas as pd, numpy as np, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

BASE  = Path(__file__).parent.parent.parent
CACHE = BASE / 'data_cache'

from hybrid.codebear.hybrid_v5_dynbtceth import (
    load_series, run_crypto_v5_daily, calc_metrics
)


def walk_forward(equity, is_end='2020-12-31', oos_start='2021-01-01', rf=0.04):
    """IS/OOS Walk-Forward"""
    eq = equity.dropna()
    is_eq  = eq[eq.index <= pd.Timestamp(is_end)]
    oos_eq = eq[eq.index >= pd.Timestamp(oos_start)]
    if len(is_eq) < 30 or len(oos_eq) < 30:
        return dict(wf=0, is_sh=0, oos_sh=0, is_cagr=0, oos_cagr=0)

    def sh(e):
        r = e.pct_change().dropna().values
        ex = r - rf/252
        return float(np.mean(ex)/np.std(ex)*np.sqrt(252)) if np.std(ex) > 0 else 0.0

    def cg(e):
        y = len(e)/252
        return float((e.iloc[-1]/e.iloc[0])**(1/y)-1) if y > 0 else 0.0

    is_s = sh(is_eq); oos_s = sh(oos_eq)
    wf = oos_s / is_s if is_s > 0 else 0.0
    return dict(wf=round(wf,3), is_sh=round(is_s,3), oos_sh=round(oos_s,3),
                is_cagr=round(cg(is_eq),4), oos_cagr=round(cg(oos_eq),4))


def build_hybrid(stock_eq, crypto_eq, w_crypto):
    idx = stock_eq.index.intersection(crypto_eq.index).sort_values()
    s_r = stock_eq.loc[idx].pct_change().fillna(0)
    c_r = crypto_eq.loc[idx].pct_change().fillna(0)
    combo = (1 + (1-w_crypto)*s_r + w_crypto*c_r).cumprod()
    return combo


def main():
    print("=" * 72)
    print("🐻 Hybrid v6 — v11b 股票 + v5-D Crypto（修复前瞻偏差）")
    print("=" * 72)

    # 加载已存储的 v11b 日频净值
    print("\n[1/3] 加载 v11b 日频股票净值...")
    v4_eq = pd.read_csv(BASE/'hybrid/codebear/hybrid_v4_equity.csv',
                         index_col=0, parse_dates=True)
    stock_eq = v4_eq['StockV11b'].dropna()
    print(f"  StockV11b: {stock_eq.index[0].date()} → {stock_eq.index[-1].date()}, {len(stock_eq)} 个交易日")

    # 计算 v11b 股票独立指标
    sm = calc_metrics(stock_eq)
    swf = walk_forward(stock_eq)
    print(f"  Stock v11b: CAGR={sm['cagr']:.1%}, MaxDD={sm['max_dd']:.1%}, "
          f"Sharpe={sm['sharpe']:.2f}, WF={swf['wf']:.3f}")

    # 运行 v5-D Crypto（已修复 prev_day）
    print("\n[2/3] 运行 Crypto v5-D（修复后）...")
    btc_p = load_series('BTC_USD')
    eth_p = load_series('ETH_USD')
    gld_p = load_series('GLD')

    crypto_eq = run_crypto_v5_daily(btc_p, eth_p, gld_p, strategy='D')
    cm = calc_metrics(crypto_eq)
    cwf = walk_forward(crypto_eq)
    print(f"  Crypto v5-D: CAGR={cm['cagr']:.1%}, MaxDD={cm['max_dd']:.1%}, "
          f"Sharpe={cm['sharpe']:.2f}, WF={cwf['wf']:.3f}")

    # 参考：v9j 股票（从 v3 equity CSV 加载）
    v3_eq = pd.read_csv(BASE/'hybrid/codebear/hybrid_v3_equity.csv',
                         index_col=0, parse_dates=True)
    stock_v9j = v3_eq['StockV9j'].dropna()
    v9j_m = calc_metrics(stock_v9j)
    v9j_wf = walk_forward(stock_v9j)
    print(f"  Stock v9j (参考): CAGR={v9j_m['cagr']:.1%}, MaxDD={v9j_m['max_dd']:.1%}, "
          f"Sharpe={v9j_m['sharpe']:.2f}, WF={v9j_wf['wf']:.3f}")

    # 组合扫描
    print("\n[3/3] Hybrid 权重扫描...\n")
    w_vals = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    print(f"{'w_c':>5} | {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'WF':>6} {'IS_Sh':>7} {'OOS_Sh':>8} {'IS→OOS CAGR':>14}")
    print("-" * 82)

    best_wf_row = None
    results = []
    for wc in w_vals:
        hybrid = build_hybrid(stock_eq, crypto_eq, wc)
        m = calc_metrics(hybrid)
        wfd = walk_forward(hybrid)
        flag = "✅" if wfd['wf'] >= 0.60 else ("⚠️" if wfd['wf'] >= 0.58 else "❌")
        row = dict(wc=wc, m=m, wf=wfd)
        results.append(row)
        print(f"  {wc:>4.0%} | {m['cagr']:>6.1%} {m['max_dd']:>7.1%} "
              f"{m['sharpe']:>7.2f} {m['calmar']:>7.2f} "
              f"{wfd['wf']:>5.3f}{flag} {wfd['is_sh']:>7.2f} {wfd['oos_sh']:>8.2f} "
              f"  {wfd['is_cagr']:>5.1%}→{wfd['oos_cagr']:>5.1%}")
        if wfd['wf'] >= 0.60 and (best_wf_row is None or m['cagr'] > best_wf_row['m']['cagr']):
            best_wf_row = row

    # 与 v5-D + v9j 基准对比
    print("\n" + "=" * 72)
    print("📊 关键对比：v6 (v11b+v5D) vs v5-D (v9j+v5D)")
    print("=" * 72)

    # v5-D + v9j 参考数字（修复后验证值）
    v5d_v9j = {'w20': dict(cagr=0.483, maxdd=-0.221, wf=0.640),
               'w25': dict(cagr=0.517, maxdd=-0.251, wf=0.635)}

    if best_wf_row:
        bm = best_wf_row['m']
        bwf = best_wf_row['wf']
        bwc = best_wf_row['wc']
        ref = v5d_v9j.get(f'w{int(bwc*100)}', v5d_v9j['w20'])
        print(f"\n  v6 最优 (w={bwc:.0%}):")
        print(f"    CAGR:    {bm['cagr']:.1%}  (v5-D+v9j={ref['cagr']:.1%}, delta={bm['cagr']-ref['cagr']:+.1%})")
        print(f"    MaxDD:   {bm['max_dd']:.1%} (v5-D+v9j={ref['maxdd']:.1%}, delta={bm['max_dd']-ref['maxdd']:+.1%})")
        print(f"    WF:      {bwf['wf']:.3f}  (v5-D+v9j={ref['wf']:.3f}, delta={bwf['wf']-ref['wf']:+.3f})")

    # 保存权益曲线
    print("\n保存 v6 equity 数据...")
    idx_common = stock_eq.index.intersection(crypto_eq.index)
    save_df = pd.DataFrame({
        'StockV11b': stock_eq.loc[idx_common],
        'CryptoV5D': crypto_eq.loc[idx_common],
        'HybridV6_w20': build_hybrid(stock_eq, crypto_eq, 0.20).loc[idx_common],
        'HybridV6_w25': build_hybrid(stock_eq, crypto_eq, 0.25).loc[idx_common],
    })
    save_df.to_csv(BASE/'hybrid/codebear/hybrid_v6_equity.csv')
    print(f"  ✓ 保存到 hybrid_v6_equity.csv")


if __name__ == '__main__':
    main()
