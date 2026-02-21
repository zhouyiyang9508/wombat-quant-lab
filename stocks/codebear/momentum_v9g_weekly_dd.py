#!/usr/bin/env python3
"""
动量轮动 v9g - Weekly DD Check — 周频回撤检查版本
代码熊 🐻

核心改进：将 DD-based GLD 对冲从月频升级为周频检查
  - 月末选股（不变）
  - 每周五收盘检查回撤深度 → 下周一执行调整（T+1）
  - 目标：将真实 MaxDD 从 -26.5% 压低，保住 CAGR > 30%

严格无前瞻：
  - 周五信号基于当日收盘价（含当日收益）
  - 下周一执行（T+1），成本 0.15%
  - 月末选股信号 = 月末收盘价，T+1执行
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

GDXJ_VOL_LO_THRESH = 0.30
GDXJ_VOL_LO_FRAC   = 0.08
GDXJ_VOL_HI_THRESH = 0.45
GDXJ_VOL_HI_FRAC   = 0.18


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
        if not f.exists() or f.stat().st_size < 500: continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200:
                d[t] = df['Close'].dropna()
        except: pass
    return pd.DataFrame(d)


def compute_gld_dd_frac(dd, dd_params):
    triggered = [dd_params[th] for th in sorted(dd_params) if dd < th]
    return max(triggered) if triggered else 0.0

def apply_overlay(base_weights, spy_vol, gld_dd_frac):
    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0
    total = gdxj_v + gld_dd_frac
    if total <= 0 or not base_weights:
        return base_weights.copy()
    stock_frac = max(1.0 - total, 0.01)
    tot = sum(base_weights.values())
    if tot <= 0: return base_weights.copy()
    new = {t: w / tot * stock_frac for t, w in base_weights.items()}
    if gld_dd_frac > 0: new['GLD'] = new.get('GLD', 0) + gld_dd_frac
    if gdxj_v > 0: new['GDXJ'] = new.get('GDXJ', 0) + gdxj_v
    return new

def day_return(weights, prev_day, day, close_df, gld_p, gdx_p, gdxj_p, shy_p):
    ret = 0.0; inv = 0.0
    for t, w in weights.items():
        if t == 'GLD': s = gld_p
        elif t == 'GDX': s = gdx_p
        elif t == 'GDXJ': s = gdxj_p
        elif t == 'SHY': s = shy_p
        elif t in close_df.columns: s = close_df[t]
        else: continue
        if prev_day in s.index and day in s.index:
            p0, p1 = s.loc[prev_day], s.loc[day]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                ret += (p1 / p0 - 1) * w; inv += w
    cash = max(1.0 - inv, 0.0)
    if cash > 0.01 and shy_p is not None:
        if prev_day in shy_p.index and day in shy_p.index:
            sp, st = shy_p.loc[prev_day], shy_p.loc[day]
            if pd.notna(sp) and pd.notna(st) and sp > 0:
                ret += (st / sp - 1) * cash
    return ret


# ── Variant A: monthly DD baseline ──────────────────────────────────────

def run_monthly_dd(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                   select_fn, apply_overlays_fn, get_spy_vol_fn,
                   start='2015-01-01', end='2025-12-31', cost=0.0015):
    all_d = close_df.loc[start:end].dropna(how='all')
    me = all_d.resample('ME').last().index; td = all_d.index
    val = 1.0; peak = 1.0; ev = []; ed = []; cw = {}; pw = {}; ph = set(); pm = set()
    for i, day in enumerate(td):
        past = me[me < day]
        if len(past) > 0:
            lm = past[-1]; nxt = td[td > lm]
            if len(nxt) > 0 and day == nxt[0] and lm not in pm:
                dd = (val - peak) / peak if peak > 0 else 0
                sv = get_spy_vol_fn(sig, lm)
                w = select_fn(sig, sectors, lm, ph, gld_p, gdx_p)
                w = apply_overlays_fn(w, sv, dd)
                to = sum(abs(w.get(t, 0) - pw.get(t, 0)) for t in set(w) | set(pw)) / 2
                val *= (1 - to * cost * 2)
                cw = w.copy(); pw = w.copy()
                ph = {k for k in w if k not in ('GLD', 'GDX', 'GDXJ', 'SHY')}
                pm.add(lm)
        if i == 0: ev.append(val); ed.append(day); continue
        r = day_return(cw, td[i-1], day, close_df, gld_p, gdx_p, gdxj_p, shy_p)
        val *= (1 + r); peak = max(peak, val); ev.append(val); ed.append(day)
    return pd.Series(ev, index=pd.DatetimeIndex(ed))


# ── Variants B-K: weekly DD ─────────────────────────────────────────────

def run_weekly_dd(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                  select_fn, get_spy_vol_fn, dd_params,
                  recovery_delay_weeks=0, ratchet=False,
                  start='2015-01-01', end='2025-12-31', cost=0.0015):
    """Monthly stock selection + weekly DD overlay adjustment.

    ratchet=True: weekly overlay can only INCREASE during month.
                  Resets (re-computed fresh) at each month-end rebalance.
    recovery_delay_weeks: delay before overlay can decrease (weeks).
    """
    all_d = close_df.loc[start:end].dropna(how='all')
    td = all_d.index; me = all_d.resample('ME').last().index

    m_exec = {}
    for m in me:
        nxt = td[td > m]
        if len(nxt) > 0: m_exec[nxt[0]] = m

    w_exec = {}
    for _, grp in all_d.groupby(pd.Grouper(freq='W-FRI')):
        if len(grp) == 0: continue
        sd = grp.index[-1]; nxt = td[td > sd]
        if len(nxt) > 0:
            ed = nxt[0]
            if ed not in m_exec: w_exec[ed] = sd

    val = 1.0; peak = 1.0; ev = []; edl = []
    bw = {}; cw = {}; pw = {}; ph = set()
    spy_v = 0.15; dd_frac = 0.0; pm = set()
    dd_inc_date = None; n_wadj = 0

    for i, day in enumerate(td):
        m_today = False

        # Monthly rebalance
        if day in m_exec and m_exec[day] not in pm:
            sig_d = m_exec[day]
            dd_now = (val - peak) / peak if peak > 0 else 0
            spy_v = get_spy_vol_fn(sig, sig_d)
            bw = select_fn(sig, sectors, sig_d, ph, gld_p, gdx_p)

            ndf = compute_gld_dd_frac(dd_now, dd_params)
            # At month-end: always allow fresh computation (ratchet resets)
            # Recovery delay applies at month-end only if NOT ratchet
            if not ratchet and recovery_delay_weeks > 0 and ndf < dd_frac:
                if dd_inc_date is not None and (day - dd_inc_date).days < recovery_delay_weeks * 7:
                    ndf = dd_frac

            if ndf > dd_frac: dd_inc_date = day
            elif ndf == 0 and dd_frac > 0: dd_inc_date = None
            dd_frac = ndf
            cw = apply_overlay(bw, spy_v, dd_frac)
            to = sum(abs(cw.get(t, 0) - pw.get(t, 0)) for t in set(cw) | set(pw)) / 2
            val *= (1 - to * cost * 2)
            pw = cw.copy()
            ph = {k for k in cw if k not in ('GLD', 'GDX', 'GDXJ', 'SHY')}
            pm.add(sig_d); m_today = True

        # Weekly DD check
        if not m_today and day in w_exec and len(bw) > 0:
            dd_now = (val - peak) / peak if peak > 0 else 0
            ndf = compute_gld_dd_frac(dd_now, dd_params)

            if ratchet:
                # Ratchet: only allow increase, never decrease
                ndf = max(ndf, dd_frac)
            elif recovery_delay_weeks > 0 and ndf < dd_frac:
                if dd_inc_date is not None and (day - dd_inc_date).days < recovery_delay_weeks * 7:
                    ndf = dd_frac

            if ndf != dd_frac:
                if ndf > dd_frac: dd_inc_date = day
                elif ndf == 0 and dd_frac > 0: dd_inc_date = None
                dd_frac = ndf
                cw = apply_overlay(bw, spy_v, dd_frac)
                to = sum(abs(cw.get(t, 0) - pw.get(t, 0)) for t in set(cw) | set(pw)) / 2
                val *= (1 - to * cost * 2)
                pw = cw.copy()
                n_wadj += 1

        if i == 0: ev.append(val); edl.append(day); continue
        r = day_return(cw, td[i-1], day, close_df, gld_p, gdx_p, gdxj_p, shy_p)
        val *= (1 + r); peak = max(peak, val); ev.append(val); edl.append(day)

    return pd.Series(ev, index=pd.DatetimeIndex(edl)), n_wadj


def compute_metrics(eq, rf=0.04):
    if len(eq) < 30: return {}
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs < 0.5: return {}
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    rm = eq.cummax(); dd = (eq - rm) / rm; mdd = dd.min()
    dr = eq.pct_change().dropna()
    av = dr.std() * np.sqrt(252); ar = dr.mean() * 252
    sh = (ar - rf) / av if av > 0 else 0
    cal = cagr / abs(mdd) if mdd != 0 else 0
    comp = sh * 0.4 + cal * 0.4 + cagr * 0.2
    ddi = dd.idxmin(); dds = eq.loc[:ddi].idxmax()
    return dict(cagr=float(cagr), max_dd=float(mdd), sharpe=float(sh),
                calmar=float(cal), composite=float(comp),
                dd_start=str(dds.date()), dd_end=str(ddi.date()),
                ann_vol=float(av), final_val=float(eq.iloc[-1]))


def main():
    print("=" * 70)
    print("🐻 v9g Weekly DD Check — 周频回撤检查全面对比")
    print("=" * 70)

    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    print(f"  加载 {len(close_df.columns)} 支股票")

    v9g = load_strategy_module("v9g", Path(__file__).parent / "momentum_v9g_final.py")
    sig = v9g.precompute(close_df)
    ALL = {}
    args = (close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p)

    # ── A: Monthly DD baseline ──
    print("\n🔄 A: 月频DD基准 (v9g原始)...")
    eq_A = run_monthly_dd(*args, v9g.select, v9g.apply_overlays, v9g.get_spy_vol)
    ALL['A'] = ('月频DD v9g基准', compute_metrics(eq_A), 0)

    # ── Core 4 variants (task spec) ──
    dd_std = {-0.08: 0.30, -0.12: 0.50, -0.18: 0.60}
    dd_early = {-0.06: 0.30, -0.10: 0.50, -0.15: 0.60}

    print("🔄 B: 周频DD -8/-12/-18 → 30/50/60%...")
    eq_B, wa = run_weekly_dd(*args, v9g.select, v9g.get_spy_vol, dd_std)
    ALL['B'] = ('周频DD -8/-12/-18%', compute_metrics(eq_B), wa)

    print("🔄 C: 周频DD -6/-10/-15 → 30/50/60% (早防)...")
    eq_C, wa = run_weekly_dd(*args, v9g.select, v9g.get_spy_vol, dd_early)
    ALL['C'] = ('周频DD 早防30/50/60', compute_metrics(eq_C), wa)

    print("🔄 D: 周频DD (同B) + 2周延迟...")
    eq_D, wa = run_weekly_dd(*args, v9g.select, v9g.get_spy_vol, dd_std, recovery_delay_weeks=2)
    ALL['D'] = ('周频DD + 2周延迟', compute_metrics(eq_D), wa)

    # ── V9g original overlay strength, applied weekly ──
    dd_v9g = {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}
    print("🔄 E: 周频DD v9g原始强度 -8/-12/-18 → 40/60/70%...")
    eq_E, wa = run_weekly_dd(*args, v9g.select, v9g.get_spy_vol, dd_v9g)
    ALL['E'] = ('周频DD v9g强度', compute_metrics(eq_E), wa)

    # ── Early defense + strong overlay ──
    dd_early_strong = {-0.06: 0.40, -0.10: 0.60, -0.15: 0.70}
    print("🔄 F: 周频DD 早防 -6/-10/-15 → 40/60/70%...")
    eq_F, wa = run_weekly_dd(*args, v9g.select, v9g.get_spy_vol, dd_early_strong)
    ALL['F'] = ('周频DD 早防+强overlay', compute_metrics(eq_F), wa)

    # ── Ratchet variants (overlay only increases during month) ──
    print("🔄 H: 棘轮 + v9g强度 -8/-12/-18 → 40/60/70%...")
    eq_H, wa = run_weekly_dd(*args, v9g.select, v9g.get_spy_vol, dd_v9g, ratchet=True)
    ALL['H'] = ('棘轮 v9g强度', compute_metrics(eq_H), wa)

    print("🔄 I: 棘轮 + 早防 -6/-10/-15 → 40/60/70%...")
    eq_I, wa = run_weekly_dd(*args, v9g.select, v9g.get_spy_vol, dd_early_strong, ratchet=True)
    ALL['I'] = ('棘轮 早防+强overlay', compute_metrics(eq_I), wa)

    print("🔄 J: 棘轮 + 激进早防 -5/-8/-12 → 40/60/70%...")
    dd_aggr = {-0.05: 0.40, -0.08: 0.60, -0.12: 0.70}
    eq_J, wa = run_weekly_dd(*args, v9g.select, v9g.get_spy_vol, dd_aggr, ratchet=True)
    ALL['J'] = ('棘轮 激进早防', compute_metrics(eq_J), wa)

    # ── 4-week recovery delay ──
    print("🔄 K: 周频DD v9g强度 + 4周延迟...")
    eq_K, wa = run_weekly_dd(*args, v9g.select, v9g.get_spy_vol, dd_v9g, recovery_delay_weeks=4)
    ALL['K'] = ('周频DD v9g + 4周延迟', compute_metrics(eq_K), wa)

    # ── Very aggressive: max defense at first sign ──
    print("🔄 L: 周频DD -5/-10/-15 → 50/70/80%...")
    dd_ultra = {-0.05: 0.50, -0.10: 0.70, -0.15: 0.80}
    eq_L, wa = run_weekly_dd(*args, v9g.select, v9g.get_spy_vol, dd_ultra, ratchet=True)
    ALL['L'] = ('棘轮 超激进防御', compute_metrics(eq_L), wa)

    # ── Ratchet + early + 30/50/60 (lighter) ──
    print("🔄 M: 棘轮 + 早防 -6/-10/-15 → 30/50/60%...")
    eq_M, wa = run_weekly_dd(*args, v9g.select, v9g.get_spy_vol, dd_early, ratchet=True)
    ALL['M'] = ('棘轮 早防轻overlay', compute_metrics(eq_M), wa)

    # ── Print results ──
    print("\n" + "=" * 100)
    print("📊 完整对比表格（日频权益曲线, 2015-2025）")
    print("=" * 100)
    hdr = f"{'ID':<3} {'描述':<24} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Comp':>8} {'周调':>4} {'DD区间'}"
    print(hdr); print("-" * 100)

    best_k = 'A'; best_comp = -999
    for k in sorted(ALL.keys()):
        lab, m, wa = ALL[k]
        c = m.get('composite', 0)
        if c > best_comp: best_comp = c; best_k = k
        tag = ' ★基准' if k == 'A' else ''
        dd_r = f"{m.get('dd_start','?')} → {m.get('dd_end','?')}"
        print(f" {k}  {lab:<24} {m['cagr']:>6.1%} {m['max_dd']:>7.1%} "
              f"{m['sharpe']:>7.2f} {m['calmar']:>7.2f} {c:>8.4f} {wa:>4} {dd_r}{tag}")

    print("-" * 100)
    # Also find best MaxDD with CAGR > 30%
    best_dd_k = None; best_dd_val = 0
    for k in sorted(ALL.keys()):
        lab, m, wa = ALL[k]
        if m['cagr'] > 0.30 and m['max_dd'] > best_dd_val:
            best_dd_val = m['max_dd']; best_dd_k = k

    blab, bm, bwa = ALL[best_k]
    print(f"\n🏆 最优Composite: {best_k} ({blab})")
    print(f"   Composite {bm['composite']:.4f} vs A {ALL['A'][1]['composite']:.4f}")

    if best_dd_k and best_dd_k != best_k:
        ddl, ddm, ddw = ALL[best_dd_k]
        print(f"\n🛡️ 最优MaxDD (CAGR>30%): {best_dd_k} ({ddl})")
        print(f"   MaxDD {ddm['max_dd']:.1%} CAGR {ddm['cagr']:.1%} Composite {ddm['composite']:.4f}")

    # ── IS/OOS analysis for best weekly variant (K) and baseline (A) ──
    print("\n" + "=" * 70)
    print("📊 IS/OOS 分析 (IS=2015-2020, OOS=2021-2025)")
    print("=" * 70)

    # A baseline IS/OOS
    eq_A_IS = run_monthly_dd(*args, v9g.select, v9g.apply_overlays, v9g.get_spy_vol,
                              start='2015-01-01', end='2020-12-31')
    eq_A_OOS = run_monthly_dd(*args, v9g.select, v9g.apply_overlays, v9g.get_spy_vol,
                               start='2021-01-01', end='2025-12-31')
    m_A_IS = compute_metrics(eq_A_IS); m_A_OOS = compute_metrics(eq_A_OOS)

    # K IS/OOS
    eq_K_IS, _ = run_weekly_dd(*args, v9g.select, v9g.get_spy_vol, dd_v9g,
                                recovery_delay_weeks=4,
                                start='2015-01-01', end='2020-12-31')
    eq_K_OOS, _ = run_weekly_dd(*args, v9g.select, v9g.get_spy_vol, dd_v9g,
                                 recovery_delay_weeks=4,
                                 start='2021-01-01', end='2025-12-31')
    m_K_IS = compute_metrics(eq_K_IS); m_K_OOS = compute_metrics(eq_K_OOS)

    # F IS/OOS (second best)
    eq_F_IS, _ = run_weekly_dd(*args, v9g.select, v9g.get_spy_vol, dd_early_strong,
                                start='2015-01-01', end='2020-12-31')
    eq_F_OOS, _ = run_weekly_dd(*args, v9g.select, v9g.get_spy_vol, dd_early_strong,
                                 start='2021-01-01', end='2025-12-31')
    m_F_IS = compute_metrics(eq_F_IS); m_F_OOS = compute_metrics(eq_F_OOS)

    for lbl, mis, moos in [('A (月频基准)', m_A_IS, m_A_OOS),
                            ('K (周频4w延迟)', m_K_IS, m_K_OOS),
                            ('F (早防+强overlay)', m_F_IS, m_F_OOS)]:
        wf = moos['sharpe'] / mis['sharpe'] if mis.get('sharpe', 0) > 0 else 0
        print(f"\n  {lbl}:")
        print(f"    IS  Sharpe={mis['sharpe']:.2f}  CAGR={mis['cagr']:.1%}  MaxDD={mis['max_dd']:.1%}")
        print(f"    OOS Sharpe={moos['sharpe']:.2f}  CAGR={moos['cagr']:.1%}  MaxDD={moos['max_dd']:.1%}")
        print(f"    WF ratio = {wf:.2f}")

    # Save
    out = {}
    for k in sorted(ALL.keys()):
        lab, m, wa = ALL[k]
        out[k] = {'label': lab, 'metrics': m, 'weekly_adj': wa}
    out['best_composite'] = best_k
    out['best_maxdd_cagr30'] = best_dd_k
    out['is_oos'] = {
        'A': {'is': m_A_IS, 'oos': m_A_OOS},
        'K': {'is': m_K_IS, 'oos': m_K_OOS},
        'F': {'is': m_F_IS, 'oos': m_F_OOS},
    }

    jf = Path(__file__).parent / "momentum_v9g_weekly_dd_results.json"
    jf.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n💾 Results → {jf}")
    return ALL, best_k


if __name__ == '__main__':
    main()
