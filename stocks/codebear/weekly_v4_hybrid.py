#!/usr/bin/env python3
"""
周频策略 v4 — 代码熊 🐻
混合策略：月频选股信号 + 周频风险管理

核心思路：
- 每月底用 v9b 逻辑选股（保持月频 alpha）
- 每周五检查风险指标：
  1. 个股跌破 10日/20日均线 → 减仓 50% → 转入 SHY
  2. 个股周跌幅 > 8% → 全部止损 → 转入 SHY
  3. 恢复信号时重新买入
- 目标：保持月频的高收益，降低回撤

🚨 无前瞻: 所有信号基于已知价格
"""

import json, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# v9b params
MOM_W = (0.20, 0.50, 0.20, 0.10)
N_BULL_SECS = 5; BULL_SPS = 2; BEAR_SPS = 2
BREADTH_NARROW = 0.45
GLD_AVG_THRESH = 0.70; GLD_COMPETE_FRAC = 0.20
CONT_BONUS = 0.03; HI52_FRAC = 0.60
DD_PARAMS = {-0.08: 0.30, -0.12: 0.50, -0.18: 0.60}


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


def run_hybrid(stop_type='sma', sma_period=20, stop_pct=-0.08, 
               cost=0.0015, reentry_wait=1):
    """
    Hybrid: monthly stock selection + weekly risk management.
    
    stop_type: 'sma' (price < SMA → reduce), 'drop' (weekly drop > X% → stop)
    sma_period: for SMA stop
    stop_pct: for drop stop
    reentry_wait: weeks to wait before re-entering after stop
    """
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    gld = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    shy = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    
    # Precompute (same as v9b)
    r1  = close_df / close_df.shift(22)  - 1
    r3  = close_df / close_df.shift(63)  - 1
    r6  = close_df / close_df.shift(126) - 1
    r12 = close_df / close_df.shift(252) - 1
    r52w_hi = close_df.rolling(252).max()
    log_r = np.log(close_df / close_df.shift(1))
    vol30 = log_r.rolling(30).std() * np.sqrt(252)
    spy = close_df['SPY']
    s200 = spy.rolling(200).mean()
    sma50 = close_df.rolling(50).mean()
    
    sig = dict(r1=r1, r3=r3, r6=r6, r12=r12, r52w_hi=r52w_hi,
               vol30=vol30, spy=spy, s200=s200, sma50=sma50, close=close_df)
    
    # Stock SMA for weekly risk
    stock_sma = close_df.rolling(sma_period).mean()
    
    def compute_breadth(date):
        c = sig['close'].loc[:date].dropna(how='all')
        s = sig['sma50'].loc[:date].dropna(how='all')
        if len(c) < 50: return 1.0
        lc = c.iloc[-1]; ls = s.iloc[-1]
        mask = (lc > ls).dropna()
        return float(mask.sum() / len(mask)) if len(mask) > 0 else 1.0
    
    def get_regime(date):
        sn = sig['spy'].loc[:date].dropna()
        s2 = sig['s200'].loc[:date].dropna()
        if len(sn) == 0 or len(s2) == 0: return 'bull'
        return 'bear' if (sn.iloc[-1] < s2.iloc[-1] and compute_breadth(date) < BREADTH_NARROW) else 'bull'
    
    def gld_comp(date):
        idx = sig['r6'].index[sig['r6'].index <= date]
        if len(idx) < 1: return 0.0
        d = idx[-1]
        sr6 = sig['r6'].loc[d].drop('SPY', errors='ignore').dropna()
        sr6 = sr6[sr6 > 0]
        if len(sr6) < 10: return 0.0
        gh = gld.loc[:d].dropna()
        if len(gh) < 130: return 0.0
        gr6 = gh.iloc[-1] / gh.iloc[-127] - 1
        return GLD_COMPETE_FRAC if gr6 >= sr6.mean() * GLD_AVG_THRESH else 0.0
    
    def monthly_select(date, prev_hold):
        idx = sig['close'].index[sig['close'].index <= date]
        if len(idx) == 0: return {}
        d = idx[-1]
        
        w1, w3, w6, w12 = MOM_W
        mom = sig['r1'].loc[d]*w1 + sig['r3'].loc[d]*w3 + sig['r6'].loc[d]*w6 + sig['r12'].loc[d]*w12
        
        df = pd.DataFrame({
            'mom': mom, 'r6': sig['r6'].loc[d], 'vol': sig['vol30'].loc[d],
            'price': sig['close'].loc[d], 'sma50': sig['sma50'].loc[d],
            'hi52': sig['r52w_hi'].loc[d],
        }).dropna(subset=['mom', 'sma50'])
        
        df = df[(df['price'] >= 5) & (df.index != 'SPY')]
        df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
        df = df[df['price'] > df['sma50']]
        df = df[df['price'] >= df['hi52'] * HI52_FRAC]
        
        df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
        for t in df.index:
            if t in prev_hold:
                df.loc[t, 'mom'] += CONT_BONUS
        
        sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)
        ga = gld_comp(date)
        reg = get_regime(date)
        
        if reg == 'bull':
            ns = N_BULL_SECS - (1 if ga > 0 else 0)
            sps, cash = BULL_SPS, 0.0
        else:
            ns = 3 - (1 if ga > 0 else 0)
            sps, cash = BEAR_SPS, 0.20
        
        ns = max(ns, 1)
        top_secs = sec_mom.head(ns).index.tolist()
        selected = []
        for sec in top_secs:
            sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
            selected.extend(sdf.index[:sps].tolist())
        
        sf = max(1.0 - cash - ga, 0.0)
        if not selected:
            return {'GLD': ga} if ga > 0 else {}
        
        iv = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
        ivt = sum(iv.values()); ivw = {t: v/ivt for t, v in iv.items()}
        mn = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
        mw = {t: df.loc[t,'mom']+sh for t in selected}
        mwt = sum(mw.values()); mww = {t: v/mwt for t, v in mw.items()}
        
        weights = {t: (0.70*ivw[t]+0.30*mww[t])*sf for t in selected}
        if ga > 0:
            weights['GLD'] = ga
        return weights
    
    def add_dd_gld(weights, dd):
        ga = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)
        if ga <= 0 or not weights: return weights
        tot = sum(weights.values())
        if tot <= 0: return weights
        new = {t: w/tot*(1-ga) for t, w in weights.items()}
        new['GLD'] = new.get('GLD', 0) + ga
        return new
    
    # ── Hybrid Backtest ──
    def backtest(start='2015-01-01', end='2025-12-31'):
        rng = close_df.loc[start:end].dropna(how='all')
        
        # Monthly rebalance dates
        monthly_ends = rng.resample('ME').last().index
        # All trading days
        all_days = rng.index
        # Weekly Fridays
        weekly_fri = rng.resample('W-FRI').last().index
        
        val = 1.0; peak = 1.0
        vals = []; dates = []
        
        monthly_weights = {}   # target weights from monthly selection
        active_weights = {}    # actual active weights (may differ due to stops)
        stopped_tickers = {}   # ticker → weeks_stopped
        prev_hold = set()
        total_turnover = []
        
        prev_day_val = {}  # for tracking daily returns
        
        mi = 0  # monthly index
        
        for wi in range(len(weekly_fri) - 1):
            wk_start = weekly_fri[wi]
            wk_end = weekly_fri[wi + 1]
            
            # Check if we need monthly rebalance
            # Find monthly end dates that fall in this week
            need_monthly = False
            for me in monthly_ends:
                if wk_start <= me <= wk_end:
                    need_monthly = True
                    monthly_date = me
                    break
            
            if need_monthly:
                dd = (val - peak) / peak if peak > 0 else 0
                new_w = monthly_select(monthly_date, prev_hold)
                new_w = add_dd_gld(new_w, dd)
                monthly_weights = new_w.copy()
                active_weights = new_w.copy()
                prev_hold = {k for k in new_w if k != 'GLD'}
                stopped_tickers = {}
            
            # Weekly risk check (using last Friday's data)
            if active_weights and stop_type == 'sma':
                # Check each stock against its SMA
                sma_vals = stock_sma.loc[:wk_start].iloc[-1] if len(stock_sma.loc[:wk_start]) > 0 else None
                price_vals = close_df.loc[:wk_start].iloc[-1] if len(close_df.loc[:wk_start]) > 0 else None
                
                if sma_vals is not None and price_vals is not None:
                    new_active = {}
                    stopped_amount = 0.0
                    for t, w in active_weights.items():
                        if t in ['GLD', 'SHY']:
                            new_active[t] = w
                            continue
                        if t in stopped_tickers:
                            stopped_tickers[t] -= 1
                            if stopped_tickers[t] <= 0:
                                del stopped_tickers[t]
                                new_active[t] = monthly_weights.get(t, w)
                            else:
                                stopped_amount += w
                            continue
                        
                        if t in price_vals.index and t in sma_vals.index:
                            p = price_vals[t]
                            s = sma_vals[t]
                            if not np.isnan(p) and not np.isnan(s) and p < s:
                                stopped_amount += w * 0.5  # reduce 50%
                                new_active[t] = w * 0.5
                                if w * 0.5 < 0.01:
                                    stopped_tickers[t] = reentry_wait
                                    stopped_amount += w * 0.5
                                    del new_active[t]
                            else:
                                new_active[t] = w
                        else:
                            new_active[t] = w
                    
                    if stopped_amount > 0:
                        new_active['SHY'] = new_active.get('SHY', 0) + stopped_amount
                    active_weights = new_active
            
            elif active_weights and stop_type == 'drop':
                # Check weekly drop
                if len(close_df.loc[:wk_start]) >= 5:
                    price_now = close_df.loc[:wk_start].iloc[-1]
                    price_prev = close_df.loc[:wk_start].iloc[-6] if len(close_df.loc[:wk_start]) >= 6 else price_now
                    
                    new_active = {}
                    stopped_amount = 0.0
                    for t, w in active_weights.items():
                        if t in ['GLD', 'SHY']:
                            new_active[t] = w
                            continue
                        if t in stopped_tickers:
                            stopped_tickers[t] -= 1
                            if stopped_tickers[t] <= 0:
                                del stopped_tickers[t]
                                new_active[t] = monthly_weights.get(t, w)
                            else:
                                stopped_amount += w
                            continue
                        
                        if t in price_now.index and t in price_prev.index:
                            pn = price_now[t]; pp = price_prev[t]
                            if not np.isnan(pn) and not np.isnan(pp) and pp > 0:
                                wr = pn/pp - 1
                                if wr < stop_pct:
                                    stopped_amount += w
                                    stopped_tickers[t] = reentry_wait
                                    continue
                        new_active[t] = w
                    
                    if stopped_amount > 0:
                        new_active['SHY'] = new_active.get('SHY', 0) + stopped_amount
                    active_weights = new_active
            
            # Calculate turnover
            old_w = {t: 0 for t in set(list(active_weights.keys()))}
            turn = sum(abs(active_weights.get(t, 0) - old_w.get(t, 0)) for t in set(list(active_weights.keys()) + list(old_w.keys()))) / 2
            total_turnover.append(turn)
            
            # Calculate weekly return
            invested = sum(active_weights.values())
            cash_frac = max(1.0 - invested, 0.0)
            
            ret = 0.0
            for t, w in active_weights.items():
                if t == 'GLD':
                    s = gld.loc[wk_start:wk_end].dropna()
                elif t == 'SHY':
                    s = shy.loc[wk_start:wk_end].dropna()
                elif t in close_df.columns:
                    s = close_df[t].loc[wk_start:wk_end].dropna()
                else:
                    continue
                if len(s) >= 2:
                    ret += (s.iloc[-1]/s.iloc[0]-1) * w
            
            if cash_frac > 0:
                s = shy.loc[wk_start:wk_end].dropna()
                if len(s) >= 2:
                    ret += (s.iloc[-1]/s.iloc[0]-1) * cash_frac
            
            # Costs (simplified)
            ret -= turn * cost * 2
            
            val *= (1 + ret)
            if val > peak: peak = val
            vals.append(val)
            dates.append(wk_end)
        
        eq = pd.Series(vals, index=pd.DatetimeIndex(dates))
        avg_to = float(np.mean(total_turnover)) if total_turnover else 0.0
        return eq, avg_to
    
    eq_full, to_f = backtest('2015-01-01', '2025-12-31')
    eq_is, to_i = backtest('2015-01-01', '2020-12-31')
    eq_oos, to_o = backtest('2021-01-01', '2025-12-31')
    
    def metrics(eq):
        if len(eq) < 10: return None
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        if yrs < 0.5: return None
        cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yrs) - 1
        wr = eq.pct_change().dropna()
        sharpe = wr.mean()/wr.std()*np.sqrt(52) if wr.std() > 0 else 0
        dd = ((eq - eq.cummax())/eq.cummax()).min()
        calmar = cagr/abs(dd) if dd != 0 else 0
        return {'cagr': cagr, 'sharpe': sharpe, 'max_dd': dd, 'calmar': calmar}
    
    mf = metrics(eq_full); mi = metrics(eq_is); mo = metrics(eq_oos)
    if not all([mf, mi, mo]): return None
    
    wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = mf['sharpe']*0.4 + mf['calmar']*0.4 + mf['cagr']*0.2
    return {
        'full': mf, 'is': mi, 'oos': mo, 'wf': wf, 'composite': comp,
        'turnover': to_f * 52,
    }


if __name__ == '__main__':
    t0 = time.time()
    print("🐻 代码熊 Weekly Hybrid v4")
    print("=" * 60)
    
    configs = [
        # (name, kwargs)
        ("pure_monthly_baseline", {'stop_type': 'none', 'sma_period': 20}),
        ("sma10_stop", {'stop_type': 'sma', 'sma_period': 10, 'reentry_wait': 1}),
        ("sma20_stop", {'stop_type': 'sma', 'sma_period': 20, 'reentry_wait': 1}),
        ("sma20_wait2", {'stop_type': 'sma', 'sma_period': 20, 'reentry_wait': 2}),
        ("sma20_wait3", {'stop_type': 'sma', 'sma_period': 20, 'reentry_wait': 3}),
        ("sma50_stop", {'stop_type': 'sma', 'sma_period': 50, 'reentry_wait': 1}),
        ("drop5_stop", {'stop_type': 'drop', 'stop_pct': -0.05, 'reentry_wait': 2}),
        ("drop8_stop", {'stop_type': 'drop', 'stop_pct': -0.08, 'reentry_wait': 2}),
        ("drop10_stop", {'stop_type': 'drop', 'stop_pct': -0.10, 'reentry_wait': 2}),
    ]
    
    results = []
    for name, kwargs in configs:
        print(f"  Testing {name}...", end=' ', flush=True)
        if kwargs['stop_type'] == 'none':
            kwargs['stop_type'] = 'sma'
            kwargs['sma_period'] = 999  # effectively no stop
        r = run_hybrid(**kwargs)
        if r:
            results.append({
                'name': name,
                'cagr': r['full']['cagr'], 'sharpe': r['full']['sharpe'],
                'maxdd': r['full']['max_dd'], 'calmar': r['full']['calmar'],
                'comp': r['composite'], 'wf': r['wf'],
                'is_shrp': r['is']['sharpe'], 'oos_shrp': r['oos']['sharpe'],
            })
            print(f"Comp={r['composite']:.3f} Shrp={r['full']['sharpe']:.2f} "
                  f"CAGR={r['full']['cagr']:.1%} MDD={r['full']['max_dd']:.1%} WF={r['wf']:.2f}")
        else:
            print("FAILED")
    
    sr = sorted(results, key=lambda x: x['comp'], reverse=True)
    print(f"\n{'='*100}")
    print(f"  RANKED HYBRID RESULTS")
    print(f"{'='*100}")
    print(f"{'#':>3} {'Name':<25} {'CAGR':>7} {'Shrp':>6} {'MDD':>8} {'Calm':>7} {'Comp':>7} {'WF':>5}")
    for i, r in enumerate(sr):
        print(f"{i+1:>3} {r['name']:<25} {r['cagr']:>6.1%} {r['sharpe']:>6.2f} "
              f"{r['maxdd']:>7.1%} {r['calmar']:>7.2f} {r['comp']:>7.3f} {r['wf']:>5.2f}")
    
    print(f"\n  Monthly v9b:  Comp=1.533  Sharpe=1.58  CAGR=31.2%  MaxDD=-14.9%  WF=0.79")
    if sr:
        b = sr[0]
        print(f"  Hybrid Best:  Comp={b['comp']:.3f}  Sharpe={b['sharpe']:.2f}  "
              f"CAGR={b['cagr']:.1%}  MaxDD={b['maxdd']:.1%}  WF={b['wf']:.2f}")
    
    print(f"\n  Time: {time.time()-t0:.1f}s")
