"""
Momentum v12c — 激进模式实验
================================
基于 v11b，测试四个进攻性调整（大袋熊要求）：
  ① VOL_TARGET_ANN: 0.11 → 0.25（别老减仓）
  ② DEFENSIVE_FRAC: 0.12 → 0.0（删掉软牛防御桥）
  ③ 权重: 70%vol+30%mom → 20%vol+80%mom（进攻优先）
  ④ N_BULL_SECS: 5/4 → 2/2（牛市只选 Top 2 行业）

代码熊 🐻 | 2026-02-22
"""

import pandas as pd, numpy as np, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import stocks.codebear.momentum_v11b_final as v11b

CACHE = Path(__file__).parent.parent.parent / 'data_cache'

# ─── 工具 ────────────────────────────────────────────────────────────────────
def lc(name):
    fp = CACHE / f'{name}.csv'
    if not fp.exists(): return pd.Series(dtype=float)
    df = pd.read_csv(fp); c = 'Date' if 'Date' in df.columns else df.columns[0]
    df[c] = pd.to_datetime(df[c])
    return df.set_index(c).sort_index()['Close'].dropna()

def compute_wf(eq, is_end='2021-12-31', oos_start='2022-01-01', rf=0.04):
    """IS(2015-2021) / OOS(2022-2025) WF"""
    is_eq  = eq[eq.index <= pd.Timestamp(is_end)]
    oos_eq = eq[eq.index >= pd.Timestamp(oos_start)]
    if len(is_eq) < 12 or len(oos_eq) < 12: return 0.0, 0.0, 0.0
    def sh(e):
        r = e.pct_change().dropna()
        return float((r.mean()-rf/12)/r.std()*np.sqrt(12)) if r.std()>0 else 0.0
    i, o = sh(is_eq), sh(oos_eq)
    return round(o/i, 3) if i > 0 else 0.0, round(i,3), round(o,3)

def run_experiment(patches: dict, label: str,
                   close_df, sig, sectors,
                   gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices):
    """patch v11b 模块级常量后跑回测，完成后还原"""
    orig = {k: getattr(v11b, k) for k in patches}
    for k, val in patches.items():
        setattr(v11b, k, val)
    try:
        eq, avg_to, rh, bh = v11b.run_backtest(
            close_df, sig, sectors,
            gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices)
    finally:
        for k, val in orig.items():
            setattr(v11b, k, val)
    m  = v11b.compute_metrics(eq)
    wf, is_sh, oos_sh = compute_wf(eq)
    total = sum(rh.values())
    return dict(label=label, m=m, wf=wf, is_sh=is_sh, oos_sh=oos_sh,
                regime=rh, avg_to=avg_to, total_months=total)


def patched_select_aggressive(sig, sectors, date, prev_hold,
                               gld_p, gdx_p, tlt_p, ief_p, def_prices):
    """
    同 v11b.select，但：
      - 权重 20%vol + 80%mom（原 70%vol + 30%mom）
      - 防御桥已通过 DEFENSIVE_FRAC=0 禁用（模块级常量覆盖即可）
    """
    # ── 复用 v11b 内部大部分逻辑，只改权重行 ──────────────────────────────
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}, 'bull_hi', None
    d = idx[-1]

    w1, w3, w6, w12 = v11b.MOM_W
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
    df = df[df['price'] >= df['hi52'] * v11b.HI52_FRAC]
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    df = df[df['sector'] != 'Unknown']
    for t in df.index:
        if t in prev_hold: df.loc[t, 'mom'] += v11b.CONT_BONUS
    if len(df) == 0: return {}, 'bull_hi', None

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    gld_a = v11b.asset_compete(sig, date, gld_p, v11b.GLD_AVG_THRESH, v11b.GLD_COMPETE_FRAC)
    gdx_a = v11b.asset_compete(sig, date, gdx_p, v11b.GDX_AVG_THRESH, v11b.GDX_COMPETE_FRAC)
    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)

    reg, breadth = v11b.get_three_regime(sig, date)
    bond_ticker, bond_frac = v11b.select_best_bond(tlt_p, ief_p, date)
    use_bond = (reg == 'bear' and bond_frac > 0)
    actual_bond_frac = bond_frac if use_bond else 0.0

    # 防御桥：DEFENSIVE_FRAC 已被 patch 为 0，所以 def_frac = 0
    if reg == 'soft_bull' and v11b.DEFENSIVE_FRAC > 0:
        avail_def = [t for t, p in def_prices.items()
                     if p is not None and len(p.loc[:date].dropna()) > 0]
        def_alloc = {t: v11b.DEFENSIVE_FRAC / len(v11b.DEFENSIVE_ETFS) for t in avail_def}
        def_frac = sum(def_alloc.values())
    else:
        avail_def = []; def_alloc = {}; def_frac = 0.0

    if reg == 'bull_hi':
        n_secs = max(v11b.N_BULL_SECS_HI - n_compete, 1)
        sps, bear_cash = v11b.BULL_SPS, 0.0
    elif reg == 'soft_bull':
        n_secs = max(v11b.N_BULL_SECS - n_compete, 1)
        sps, bear_cash = v11b.BULL_SPS, 0.0
    else:
        n_secs = max(3 - n_compete, 1)
        sps = v11b.BEAR_SPS
        bear_cash = max(0.20 - actual_bond_frac, 0.0)

    top_secs = sec_mom.head(n_secs).index.tolist()
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    stock_frac = max(1.0 - bear_cash - total_compete - actual_bond_frac - def_frac, 0.0)

    if not selected:
        w = {}
        if gld_a > 0: w['GLD'] = gld_a
        if gdx_a > 0: w['GDX'] = gdx_a
        if use_bond and bond_ticker: w[bond_ticker] = actual_bond_frac
        w.update(def_alloc)
        return w, reg, bond_ticker

    iv   = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw   = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}

    # ← 核心改动：80% Momentum / 20% Vol（原 30%/70%）
    weights = {t: (0.20*iv_w[t] + 0.80*mw_w[t]) * stock_frac for t in selected}

    if gld_a > 0: weights['GLD'] = weights.get('GLD', 0) + gld_a
    if gdx_a > 0: weights['GDX'] = weights.get('GDX', 0) + gdx_a
    if use_bond and bond_ticker: weights[bond_ticker] = actual_bond_frac
    weights.update(def_alloc)
    return weights, reg, bond_ticker


def main():
    print("=" * 72)
    print("🐻 v12c — 激进模式实验 (大袋熊四连击)")
    print("=" * 72)

    print("\n[1/3] 加载数据...")
    with open(CACHE / 'sp500_tickers.txt') as f:
        tickers = [t.strip() for t in f if t.strip()]
    with open(CACHE / 'sp500_sectors.json') as f:
        sectors = json.load(f)

    close_df = v11b.load_stocks(tickers + ['SPY'])
    stock_cols = [c for c in close_df.columns if c not in ('GLD','GDX','GDXJ','SHY','TLT','IEF')]

    gld_p  = lc('GLD'); gdx_p  = lc('GDX'); gdxj_p = lc('GDXJ')
    shy_p  = lc('SHY'); tlt_p  = lc('TLT'); ief_p  = lc('IEF')
    def_prices = {'XLV': lc('XLV'), 'XLP': lc('XLP'), 'XLU': lc('XLU')}

    print("[2/3] 预计算信号...")
    sig = v11b.precompute(close_df[stock_cols])

    print("[3/3] 运行实验...\n")

    def run(label, patches, use_aggressive_select=False):
        orig_select = None
        if use_aggressive_select:
            orig_select = v11b.select
            v11b.select = lambda sig, sectors, date, prev_hold, gld_p, gdx_p, tlt_p, ief_p, def_prices: \
                patched_select_aggressive(sig, sectors, date, prev_hold, gld_p, gdx_p, tlt_p, ief_p, def_prices)
        result = run_experiment(patches, label, close_df[stock_cols], sig, sectors,
                                gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices)
        if orig_select:
            v11b.select = orig_select
        return result

    configs = [
        # (label, patches, use_aggressive_select)
        ("Baseline v11b",    {}, False),
        ("① VOL_TARGET=25%", {'VOL_TARGET_ANN': 0.25}, False),
        ("② 删防御桥",        {'DEFENSIVE_FRAC': 0.0}, False),
        ("③ 80%Mom/20%Vol",  {}, True),   # select function patched
        ("④ Top2行业",        {'N_BULL_SECS': 2, 'N_BULL_SECS_HI': 2}, False),
        ("全部叠加",           {'VOL_TARGET_ANN': 0.25,
                               'DEFENSIVE_FRAC': 0.0,
                               'N_BULL_SECS': 2,
                               'N_BULL_SECS_HI': 2}, True),
    ]

    rows = []
    for label, patches, agg_select in configs:
        print(f"  [{label}]...")
        r = run(label, patches, agg_select)
        rows.append(r)

    # 输出
    print("\n" + "=" * 78)
    print("📊 结果对比 (月频，IS=2015-2021，OOS=2022-2025)")
    print("=" * 78)
    print(f"{'策略':>18} | {'CAGR':>6} {'MaxDD':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'WF':>6} {'IS_Sh':>6} {'OOS_Sh':>7} {'换手':>5}")
    print("-" * 78)

    base = rows[0]['m']
    for r in rows:
        m = r['m']
        flag = "✅" if r['wf'] >= 0.70 else ("⚠️" if r['wf'] >= 0.60 else "❌")
        cagr_d = f" ({m['cagr']-base['cagr']:+.1%})" if r['label'] != 'Baseline v11b' else ""
        print(f"  {r['label']:>16} | {m['cagr']:>5.1%}{cagr_d:<7} {m['max_dd']:>7.1%} "
              f"{m['sharpe']:>7.2f} {m['calmar']:>7.2f} "
              f"{r['wf']:>5.3f}{flag} {r['is_sh']:>6.2f} {r['oos_sh']:>7.2f} "
              f"{r['avg_to']:>5.1%}")

    print("\n📐 Regime 分布（防御桥删除后 soft_bull 比例变化）:")
    for r in rows:
        rh = r['regime']; total = sum(rh.values())
        if total > 0:
            print(f"  {r['label']:>16}: bull={rh.get('bull_hi',0)/total:.0%} "
                  f"soft={rh.get('soft_bull',0)/total:.0%} bear={rh.get('bear',0)/total:.0%}")


if __name__ == '__main__':
    main()
