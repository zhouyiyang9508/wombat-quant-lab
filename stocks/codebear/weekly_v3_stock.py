#!/usr/bin/env python3
"""
周频个股策略 v3 — 代码熊 🐻
Weekly Stock Momentum — faithful v9b port with turnover control

直接移植月频 v9b 的核心逻辑到周频:
- 多期动量：4w/13w/26w/52w (对应月频的 1m/3m/6m/12m)
- 行业轮动：top N sectors × top K stocks
- 过滤器：价格>$5, r26w>0, vol<0.65, price>SMA(50天), 52w高点过滤
- 熊市切换：SPY<SMA200+面包宽度
- GLD竞争/回撤叠加
- 持仓延续奖励
- 反向波动率+动量混合加权
- 🆕 阈值调仓（threshold rebalance）：降低周频换手

🚨 无前瞻: 周五收盘的日线数据生成信号 → 下周一执行（模拟为下周五收盘价比较）
"""

import json, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"


def load_csv(fp):
    df = pd.read_csv(fp, parse_dates=['Date'], index_col='Date')
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


def run_weekly_v9b(params):
    """Run weekly v9b-style backtest with given parameters."""
    
    # Unpack params
    MOM_W = params.get('mom_w', (0.20, 0.50, 0.20, 0.10))  # 4w, 13w, 26w, 52w
    N_BULL_SECS = params.get('n_bull_secs', 5)
    BULL_SPS = params.get('bull_sps', 2)
    BEAR_SPS = params.get('bear_sps', 2)
    BREADTH_NARROW = params.get('breadth_narrow', 0.45)
    GLD_AVG_THRESH = params.get('gld_avg_thresh', 0.70)
    GLD_COMPETE_FRAC = params.get('gld_compete_frac', 0.20)
    CONT_BONUS = params.get('cont_bonus', 0.05)  # higher than monthly to incentivize holding
    HI52_FRAC = params.get('hi52_frac', 0.60)
    USE_SHY = params.get('use_shy', True)
    COST = params.get('cost', 0.0015)
    REBAL_OVERLAP = params.get('rebal_overlap', 0.5)  # only rebalance if overlap < this
    DD_PARAMS = params.get('dd_params', {-0.08: 0.30, -0.12: 0.50, -0.18: 0.60})
    
    # Load data
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    gld = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    shy = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    
    # Precompute daily signals (same as v9b but we'll sample weekly)
    r4w  = close_df / close_df.shift(20) - 1   # ~4 weeks
    r13w = close_df / close_df.shift(65) - 1   # ~13 weeks
    r26w = close_df / close_df.shift(130) - 1  # ~26 weeks
    r52w = close_df / close_df.shift(252) - 1  # ~52 weeks (1 year)
    r52w_hi = close_df.rolling(252).max()
    log_r = np.log(close_df / close_df.shift(1))
    vol30 = log_r.rolling(30).std() * np.sqrt(252)
    spy = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200 = spy.rolling(200).mean() if spy is not None else None
    sma50 = close_df.rolling(50).mean()
    
    sig = dict(r4w=r4w, r13w=r13w, r26w=r26w, r52w=r52w, r52w_hi=r52w_hi,
               vol30=vol30, spy=spy, s200=s200, sma50=sma50, close=close_df)
    
    def compute_breadth(date):
        close = sig['close'].loc[:date].dropna(how='all')
        sma = sig['sma50'].loc[:date].dropna(how='all')
        if len(close) < 50:
            return 1.0
        lc = close.iloc[-1]; ls = sma.iloc[-1]
        mask = (lc > ls).dropna()
        return float(mask.sum() / len(mask)) if len(mask) > 0 else 1.0
    
    def get_regime(date):
        if s200 is None:
            return 'bull'
        spy_now = sig['spy'].loc[:date].dropna()
        s200_now = sig['s200'].loc[:date].dropna()
        if len(spy_now) == 0 or len(s200_now) == 0:
            return 'bull'
        spy_bear = spy_now.iloc[-1] < s200_now.iloc[-1]
        breadth_bear = compute_breadth(date) < BREADTH_NARROW
        return 'bear' if (spy_bear and breadth_bear) else 'bull'
    
    def gld_competition(date):
        idx = sig['r26w'].index[sig['r26w'].index <= date]
        if len(idx) < 1:
            return 0.0
        d = idx[-1]
        stock_r26 = sig['r26w'].loc[d].drop('SPY', errors='ignore').dropna()
        stock_r26 = stock_r26[stock_r26 > 0]
        if len(stock_r26) < 10:
            return 0.0
        avg_r26 = stock_r26.mean()
        gld_h = gld.loc[:d].dropna()
        if len(gld_h) < 130:
            return 0.0
        gld_r26 = gld_h.iloc[-1] / gld_h.iloc[-131] - 1
        return GLD_COMPETE_FRAC if gld_r26 >= avg_r26 * GLD_AVG_THRESH else 0.0
    
    def select_stocks(date, prev_hold):
        idx = sig['close'].index[sig['close'].index <= date]
        if len(idx) == 0:
            return {}
        d = idx[-1]
        
        w1, w3, w6, w12 = MOM_W
        mom = (sig['r4w'].loc[d]*w1 + sig['r13w'].loc[d]*w3 +
               sig['r26w'].loc[d]*w6 + sig['r52w'].loc[d]*w12)
        
        df = pd.DataFrame({
            'mom': mom,
            'r26w': sig['r26w'].loc[d],
            'vol': sig['vol30'].loc[d],
            'price': sig['close'].loc[d],
            'sma50': sig['sma50'].loc[d],
            'hi52': sig['r52w_hi'].loc[d],
        }).dropna(subset=['mom', 'sma50'])
        
        df = df[(df['price'] >= 5) & (df.index != 'SPY')]
        df = df[(df['r26w'] > 0) & (df['vol'] < 0.65)]
        df = df[df['price'] > df['sma50']]
        df = df[df['price'] >= df['hi52'] * HI52_FRAC]
        
        df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
        for t in df.index:
            if t in prev_hold:
                df.loc[t, 'mom'] += CONT_BONUS
        
        sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)
        gld_a = gld_competition(date)
        reg = get_regime(date)
        
        if reg == 'bull':
            n_secs = N_BULL_SECS - (1 if gld_a > 0 else 0)
            sps, cash = BULL_SPS, 0.0
        else:
            n_secs = 3 - (1 if gld_a > 0 else 0)
            sps, cash = BEAR_SPS, 0.20
        
        n_secs = max(n_secs, 1)
        top_secs = sec_mom.head(n_secs).index.tolist()
        selected = []
        for sec in top_secs:
            sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
            selected.extend(sdf.index[:sps].tolist())
        
        stock_frac = max(1.0 - cash - gld_a, 0.0)
        if not selected:
            return {'GLD': gld_a} if gld_a > 0 else {}
        
        iv = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
        iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
        mn = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
        mw = {t: df.loc[t,'mom']+sh for t in selected}
        mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}
        
        weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
        if gld_a > 0:
            weights['GLD'] = gld_a
        return weights
    
    def add_dd_gld(weights, dd):
        gld_a = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)
        if gld_a <= 0 or not weights:
            return weights
        tot = sum(weights.values())
        if tot <= 0:
            return weights
        new = {t: w/tot*(1-gld_a) for t, w in weights.items()}
        new['GLD'] = new.get('GLD', 0) + gld_a
        return new
    
    # ── Run backtest ──
    def backtest(start='2015-01-01', end='2025-12-31'):
        rng = close_df.loc[start:end].dropna(how='all')
        # Weekly Fridays
        weekly_dates = rng.resample('W-FRI').last().index
        
        vals, dates, tos = [], [], []
        prev_w, prev_h = {}, set()
        current_w = {}  # current active weights
        val = 1.0; peak = 1.0
        
        for i in range(len(weekly_dates) - 1):
            dt = weekly_dates[i]      # signal date (Friday close)
            ndt = weekly_dates[i+1]   # execution date (next Friday close)
            
            dd = (val - peak) / peak if peak > 0 else 0
            new_w = select_stocks(dt, prev_h)
            new_w = add_dd_gld(new_w, dd)
            
            # Threshold rebalancing: only rebalance if significant change
            if REBAL_OVERLAP < 1.0 and current_w:
                old_set = set(k for k in current_w if k != 'GLD')
                new_set = set(k for k in new_w if k != 'GLD')
                if old_set and new_set:
                    overlap = len(old_set & new_set) / len(old_set)
                    if overlap >= REBAL_OVERLAP:
                        # Keep current weights, just update GLD if needed
                        new_w = current_w.copy()
                        # But update GLD overlay based on new DD
                        new_w = add_dd_gld(new_w, dd)
            
            all_t = set(new_w) | set(prev_w)
            to = sum(abs(new_w.get(t,0)-prev_w.get(t,0)) for t in all_t) / 2
            tos.append(to)
            prev_w = new_w.copy()
            current_w = new_w.copy()
            prev_h = {k for k in new_w if k != 'GLD'}
            
            invested = sum(new_w.values())
            cash_frac = max(1.0 - invested, 0.0)
            
            ret = 0.0
            for t, wt in new_w.items():
                if t == 'GLD':
                    s = gld.loc[dt:ndt].dropna()
                elif t in close_df.columns:
                    s = close_df[t].loc[dt:ndt].dropna()
                else:
                    continue
                if len(s) >= 2:
                    ret += (s.iloc[-1]/s.iloc[0]-1) * wt
            
            if USE_SHY and cash_frac > 0:
                s = shy.loc[dt:ndt].dropna()
                if len(s) >= 2:
                    ret += (s.iloc[-1]/s.iloc[0]-1) * cash_frac
            
            ret -= to * COST * 2
            val *= (1 + ret)
            if val > peak: peak = val
            vals.append(val); dates.append(ndt)
        
        eq = pd.Series(vals, index=pd.DatetimeIndex(dates))
        avg_to = float(np.mean(tos)) if tos else 0.0
        return eq, avg_to
    
    # Full, IS, OOS
    eq_full, to_full = backtest('2015-01-01', '2025-12-31')
    eq_is, to_is = backtest('2015-01-01', '2020-12-31')
    eq_oos, to_oos = backtest('2021-01-01', '2025-12-31')
    
    def metrics(eq):
        if len(eq) < 10:
            return None
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        if yrs < 0.5:
            return None
        cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yrs) - 1
        # Weekly returns
        wr = eq.pct_change().dropna()
        sharpe = wr.mean()/wr.std()*np.sqrt(52) if wr.std() > 0 else 0
        dd = ((eq - eq.cummax())/eq.cummax()).min()
        calmar = cagr/abs(dd) if dd != 0 else 0
        return {'cagr': cagr, 'sharpe': sharpe, 'max_dd': dd, 'calmar': calmar}
    
    m_full = metrics(eq_full)
    m_is = metrics(eq_is)
    m_oos = metrics(eq_oos)
    
    if not all([m_full, m_is, m_oos]):
        return None
    
    wf = m_oos['sharpe'] / m_is['sharpe'] if m_is['sharpe'] > 0 else 0
    comp = m_full['sharpe']*0.4 + m_full['calmar']*0.4 + m_full['cagr']*0.2
    
    return {
        'full': m_full, 'is': m_is, 'oos': m_oos,
        'wf': wf, 'composite': comp,
        'turnover_weekly': to_full,
        'turnover_annual': to_full * 52,
    }


if __name__ == '__main__':
    t0 = time.time()
    print("🐻 代码熊 Weekly Stock Momentum v3")
    print("=" * 60)
    
    # ── Grid search over key weekly-specific params ──
    results = []
    
    configs = [
        # (name, params_override)
        # Vary continuation bonus (key for reducing turnover)
        ("base_cb0.03", {'cont_bonus': 0.03, 'rebal_overlap': 1.0}),
        ("base_cb0.05", {'cont_bonus': 0.05, 'rebal_overlap': 1.0}),
        ("base_cb0.08", {'cont_bonus': 0.08, 'rebal_overlap': 1.0}),
        ("base_cb0.10", {'cont_bonus': 0.10, 'rebal_overlap': 1.0}),
        
        # Threshold rebalancing
        ("thr50_cb0.05", {'cont_bonus': 0.05, 'rebal_overlap': 0.5}),
        ("thr50_cb0.08", {'cont_bonus': 0.08, 'rebal_overlap': 0.5}),
        ("thr60_cb0.05", {'cont_bonus': 0.05, 'rebal_overlap': 0.6}),
        ("thr60_cb0.08", {'cont_bonus': 0.08, 'rebal_overlap': 0.6}),
        ("thr70_cb0.05", {'cont_bonus': 0.05, 'rebal_overlap': 0.7}),
        ("thr70_cb0.08", {'cont_bonus': 0.08, 'rebal_overlap': 0.7}),
        ("thr70_cb0.10", {'cont_bonus': 0.10, 'rebal_overlap': 0.7}),
        ("thr80_cb0.10", {'cont_bonus': 0.10, 'rebal_overlap': 0.8}),
        
        # Different costs
        ("thr70_cb0.08_c10", {'cont_bonus': 0.08, 'rebal_overlap': 0.7, 'cost': 0.001}),
        ("thr70_cb0.08_c20", {'cont_bonus': 0.08, 'rebal_overlap': 0.7, 'cost': 0.002}),
        
        # More sectors / stocks
        ("thr70_6s3k", {'cont_bonus': 0.08, 'rebal_overlap': 0.7, 'n_bull_secs': 6, 'bull_sps': 3}),
        ("thr70_4s3k", {'cont_bonus': 0.08, 'rebal_overlap': 0.7, 'n_bull_secs': 4, 'bull_sps': 3}),
        
        # Different momentum weights (more weight on shorter-term for weekly)
        ("shortmom_thr70", {'cont_bonus': 0.08, 'rebal_overlap': 0.7, 
                            'mom_w': (0.40, 0.35, 0.15, 0.10)}),
        ("midmom_thr70", {'cont_bonus': 0.08, 'rebal_overlap': 0.7,
                          'mom_w': (0.25, 0.45, 0.20, 0.10)}),
        
        # Higher breadth threshold
        ("thr70_br50", {'cont_bonus': 0.08, 'rebal_overlap': 0.7, 'breadth_narrow': 0.50}),
        ("thr70_br40", {'cont_bonus': 0.08, 'rebal_overlap': 0.7, 'breadth_narrow': 0.40}),
    ]
    
    print(f"\nTesting {len(configs)} configurations...")
    
    for name, overrides in configs:
        params = {
            'mom_w': (0.20, 0.50, 0.20, 0.10),
            'n_bull_secs': 5, 'bull_sps': 2, 'bear_sps': 2,
            'breadth_narrow': 0.45,
            'gld_avg_thresh': 0.70, 'gld_compete_frac': 0.20,
            'cont_bonus': 0.05, 'hi52_frac': 0.60,
            'use_shy': True, 'cost': 0.0015,
            'rebal_overlap': 1.0,
            'dd_params': {-0.08: 0.30, -0.12: 0.50, -0.18: 0.60},
        }
        params.update(overrides)
        
        print(f"  Testing {name}...", end=' ', flush=True)
        r = run_weekly_v9b(params)
        
        if r:
            results.append({
                'name': name,
                'full_cagr': r['full']['cagr'],
                'full_sharpe': r['full']['sharpe'],
                'full_maxdd': r['full']['max_dd'],
                'full_calmar': r['full']['calmar'],
                'composite': r['composite'],
                'is_sharpe': r['is']['sharpe'],
                'oos_sharpe': r['oos']['sharpe'],
                'wf': r['wf'],
                'turnover': r['turnover_annual'],
                'params': params,
            })
            print(f"Comp={r['composite']:.3f} Shrp={r['full']['sharpe']:.2f} "
                  f"CAGR={r['full']['cagr']:.1%} WF={r['wf']:.2f} "
                  f"Turn={r['turnover_annual']:.1f}")
        else:
            print("FAILED")
    
    # Sort and display
    sr = sorted(results, key=lambda x: x['composite'], reverse=True)
    
    print(f"\n{'='*110}")
    print(f"  RANKED RESULTS")
    print(f"{'='*110}")
    print(f"{'#':>3} {'Name':<25} {'CAGR':>7} {'Shrp':>6} {'MDD':>8} {'Calm':>7} {'Comp':>7} {'WF':>5} {'IS':>6} {'OOS':>6} {'Turn':>6}")
    print(f"{'-'*110}")
    for i, r in enumerate(sr):
        print(f"{i+1:>3} {r['name']:<25} {r['full_cagr']:>6.1%} {r['full_sharpe']:>6.2f} "
              f"{r['full_maxdd']:>7.1%} {r['full_calmar']:>7.2f} {r['composite']:>7.3f} "
              f"{r['wf']:>5.2f} {r['is_sharpe']:>6.2f} {r['oos_sharpe']:>6.2f} "
              f"{r['turnover']:>5.1f}")
    
    # Champion detail
    if sr:
        best = sr[0]
        print(f"\n{'='*80}")
        print(f"  🏆 CHAMPION: {best['name']}")
        print(f"{'='*80}")
        print(f"  CAGR:       {best['full_cagr']:.1%}")
        print(f"  Sharpe:     {best['full_sharpe']:.2f}")
        print(f"  MaxDD:      {best['full_maxdd']:.1%}")
        print(f"  Calmar:     {best['full_calmar']:.2f}")
        print(f"  Composite:  {best['composite']:.4f}")
        print(f"  WF ratio:   {best['wf']:.2f}")
        print(f"  IS Sharpe:  {best['is_sharpe']:.2f}")
        print(f"  OOS Sharpe: {best['oos_sharpe']:.2f}")
        print(f"  Turnover:   {best['turnover']:.1f}x/yr")
    
    # vs Monthly v9b
    print(f"\n{'='*80}")
    print(f"  📊 WEEKLY STOCK v3 vs MONTHLY STOCK v9b")
    print(f"{'='*80}")
    print(f"  Monthly v9b:  Comp=1.533  Sharpe=1.58  CAGR=31.2%  MaxDD=-14.9%  WF=0.79")
    if sr:
        b = sr[0]
        print(f"  Weekly v3:    Comp={b['composite']:.3f}  Sharpe={b['full_sharpe']:.2f}  "
              f"CAGR={b['full_cagr']:.1%}  MaxDD={b['full_maxdd']:.1%}  WF={b['wf']:.2f}")
        diff = b['composite'] - 1.533
        print(f"  Composite Δ: {diff:+.3f} {'✅ BETTER' if diff > 0 else '❌ WORSE'}")
    
    # Save
    sv = [{k: v for k, v in r.items() if k != 'params'} for r in sr[:20]]
    out = Path(__file__).parent / "weekly_v3_results.json"
    with open(out, 'w') as f:
        json.dump(sv, f, indent=2, default=str)
    
    print(f"\n  Time: {time.time()-t0:.1f}s")
    print(f"  Saved to {out}")
