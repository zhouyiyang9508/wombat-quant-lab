"""
Momentum v12d — 纯动量进攻模式
================================
系统性探索：去掉 v11b 的各层防御，找到最优进攻组合

实验矩阵（逐步去壳）:
  Baseline  : v11b 原版（所有保护层）
  E1        : BULL_SPS 2→3（每行业选3只）
  E2        : VOL_TARGET 11%→20%（少减仓）
  E3        : E1 + E2
  E4        : E3 + 无 SPY 软对冲
  E5        : E4 + 无 TLT 熊市债券（最激进：纯现金对冲）
  E6        : E5 + 无 GDXJ 波动率对冲（真·纯动量）

代码熊 🐻 | 2026-02-22
"""
import pandas as pd, numpy as np, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import stocks.codebear.momentum_v11b_final as v11b

CACHE = Path(__file__).parent.parent.parent / 'data_cache'

def lc(name):
    fp = CACHE / f'{name}.csv'
    if not fp.exists(): return pd.Series(dtype=float)
    df = pd.read_csv(fp); c = 'Date' if 'Date' in df.columns else df.columns[0]
    df[c] = pd.to_datetime(df[c]); return df.set_index(c).sort_index()['Close'].dropna()

def wf(eq, is_end='2021-12-31', oos_start='2022-01-01', rf=0.04):
    is_eq  = eq[eq.index <= pd.Timestamp(is_end)]
    oos_eq = eq[eq.index >= pd.Timestamp(oos_start)]
    if len(is_eq) < 12 or len(oos_eq) < 12: return 0, 0, 0
    def sh(e):
        r = e.pct_change().dropna()
        return float((r.mean()-rf/12)/r.std()*np.sqrt(12)) if r.std()>0 else 0
    i, o = sh(is_eq), sh(oos_eq)
    return round(o/i,3) if i>0 else 0, round(i,3), round(o,3)

def make_runner(close_df, sig, sectors,
                gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices):
    """返回一个 run(patches, select_fn) 函数，捕获数据到闭包"""
    def run(patches, select_fn=None):
        orig = {k: getattr(v11b, k) for k in patches}
        orig_sel = v11b.select if select_fn else None
        for k, val in patches.items(): setattr(v11b, k, val)
        if select_fn: v11b.select = select_fn
        try:
            eq, avg_to, rh, _ = v11b.run_backtest(
                close_df, sig, sectors,
                gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices)
        finally:
            for k, val in orig.items(): setattr(v11b, k, val)
            if select_fn: v11b.select = orig_sel
        m = v11b.compute_metrics(eq)
        wfr, is_sh, oos_sh = wf(eq)
        return m, wfr, is_sh, oos_sh, avg_to, rh, eq
    return run

# ── 无 SPY 软对冲：patch apply_overlays 跳过 spy_1m ───────────────────────
def make_no_spy_overlay():
    orig = v11b.apply_overlays
    def patched(weights, spy_vol, dd, port_vol_ann, spy_1m_ret):
        return orig(weights, spy_vol, dd, port_vol_ann, 0.0)  # spy_1m=0 → 不触发
    return patched

# ── 无 TLT + 无 SPY：直接把 bear 时债券 frac 压成 0 ────────────────────────
NO_TLT_PATCHES = {'TLT_BEAR_FRAC': 0.0, 'IEF_BEAR_FRAC': 0.0}

# ── 无 GDXJ（不再用波动率触发 GDXJ）────────────────────────────────────────
NO_GDXJ_PATCHES = {'GDXJ_VOL_LO_FRAC': 0.0, 'GDXJ_VOL_HI_FRAC': 0.0}

# ────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 72)
    print("🐻 v12d — 纯动量进攻模式（逐步去壳）")
    print("=" * 72)

    print("\n[1/3] 加载数据...")
    with open(CACHE / 'sp500_tickers.txt') as f:
        tickers = [t.strip() for t in f if t.strip()]
    with open(CACHE / 'sp500_sectors.json') as f:
        sectors = json.load(f)
    close_df = v11b.load_stocks(tickers + ['SPY'])
    stock_cols = [c for c in close_df.columns if c not in ('GLD','GDX','GDXJ','SHY','TLT','IEF')]
    gld_p=lc('GLD'); gdx_p=lc('GDX'); gdxj_p=lc('GDXJ')
    shy_p=lc('SHY'); tlt_p=lc('TLT'); ief_p=lc('IEF')
    def_prices={'XLV':lc('XLV'),'XLP':lc('XLP'),'XLU':lc('XLU')}

    print("[2/3] 预计算信号...")
    sig = v11b.precompute(close_df[stock_cols])

    print("[3/3] 运行实验矩阵...\n")
    run = make_runner(close_df[stock_cols], sig, sectors,
                      gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices)

    experiments = [
        ("Baseline v11b",   {}, None),
        ("E1 SPS=3",        {'BULL_SPS': 3}, None),
        ("E2 VOL=20%",      {'VOL_TARGET_ANN': 0.20}, None),
        ("E3 SPS=3+VOL=20%",{'BULL_SPS': 3, 'VOL_TARGET_ANN': 0.20}, None),
        ("E4 +无SPY对冲",   {'BULL_SPS': 3, 'VOL_TARGET_ANN': 0.20,
                              'SPY_SOFT_HI_FRAC': 0.0}, None),
        ("E5 +无TLT债券",   {'BULL_SPS': 3, 'VOL_TARGET_ANN': 0.20,
                              'SPY_SOFT_HI_FRAC': 0.0,
                              **NO_TLT_PATCHES}, None),
        ("E6 纯动量",       {'BULL_SPS': 3, 'VOL_TARGET_ANN': 0.20,
                              'SPY_SOFT_HI_FRAC': 0.0,
                              **NO_TLT_PATCHES, **NO_GDXJ_PATCHES}, None),
    ]

    rows = []
    for label, patches, sfn in experiments:
        print(f"  [{label}]...")
        m, wfr, is_sh, oos_sh, avg_to, rh, eq = run(patches, sfn)
        rows.append((label, m, wfr, is_sh, oos_sh, avg_to, rh, eq))

    print("\n" + "=" * 80)
    print("📊 结果 (月频，IS=2015-2021，OOS=2022-2025，Composite=Sh×0.4+Cal×0.4+CAGR×0.2)")
    print("=" * 80)
    hdr = "%-18s | %6s %7s %7s %7s %6s %6s %6s %5s"
    print(hdr % ('策略','CAGR','MaxDD','Sharpe','Calmar','Comp','WF','IS→OOS','换手'))
    print("-" * 80)
    base_cagr = rows[0][1]['cagr']
    for label, m, wfr, is_sh, oos_sh, avg_to, rh, eq in rows:
        comp = m['sharpe']*0.4 + m['calmar']*0.4 + min(m['cagr'],1.0)*0.2
        flag = '✅' if wfr>=0.70 else ('⚠️' if wfr>=0.60 else '❌')
        delta = '' if label=='Baseline v11b' else ' (%+.1f%%)' % ((m['cagr']-base_cagr)*100)
        print("  %-16s | %5.1f%%%s" % (label, m['cagr']*100, delta))
        print("    MaxDD=%6.1f%%  Sh=%5.2f  Cal=%5.2f  Comp=%5.3f  WF=%5.3f%s  IS=%.2f OOS=%.2f  TO=%.1f%%" % (
            m['max_dd']*100, m['sharpe'], m['calmar'], comp,
            wfr, flag, is_sh, oos_sh, avg_to*100))

    # 最优推荐
    print("\n📌 最优配置（WF≥0.60）:")
    best = sorted([(r[2], r[1]['cagr'], r) for r in rows if r[2]>=0.60],
                   key=lambda x: x[1], reverse=True)
    if best:
        _, _, (lbl, m, wfr, is_sh, oos_sh, *_) = best[0]
        print(f"  最高 CAGR: {lbl} — CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}, WF={wfr:.3f}")

    # 保存最佳 equity 供 Hybrid 组合用
    if best:
        best_lbl, best_m, best_wfr = best[0][2][0], best[0][2][1], best[0][2][2]
        best_eq = best[0][2][7]
    else:
        best_lbl, best_eq = 'Baseline v11b', rows[0][7]
    save_path = CACHE.parent / 'hybrid/codebear/stock_v12d_aggressive_equity.csv'
    best_eq.to_csv(save_path, header=['StockV12d'])
    print(f"\n  ✓ 最佳配置 [{best_lbl}] equity 已保存 → stock_v12d_aggressive_equity.csv")
