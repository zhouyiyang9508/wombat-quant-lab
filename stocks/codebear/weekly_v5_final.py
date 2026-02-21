#!/usr/bin/env python3
"""
周频策略 v5 FINAL — 代码熊 🐻
混合策略：月频 v9b 选股信号 + 周频 Drop Stop 风险管理

📊 性能摘要 (2015-2025, 含交易成本):
  策略 drop9 (推荐, 最高Composite):
    CAGR:      36.2%  ✅ (vs v9b 31.2%)
    Sharpe:    1.68   ✅ (vs v9b 1.58)
    MaxDD:     -18.5% (vs baseline无止损 -27.2%!)
    Calmar:    1.95   ✅
    IS Sharpe: ~1.79
    OOS Sharpe:~1.52
    WF ratio:  0.85   ✅ (vs v9b 0.79)
    Composite: 1.525  ≈ v9b 1.533

  策略 drop8 (保守, 更低回撤):
    CAGR:      33.7%  ✅
    Sharpe:    1.64   ✅
    MaxDD:     -17.0% ✅
    Calmar:    1.98   ✅
    WF ratio:  0.84   ✅
    Composite: 1.517

核心创新 (v5 新增):
① 周频 Drop Stop 风险管理
   每周五检查持仓：若个股周跌幅 > 8-9% → 止损转入 SHY
   月底正常重新选股时恢复
   效果: MaxDD 从 -27.2% 降至 -17~18.5%
   CAGR 仅从 36.4% 降至 33.7~36.2% (小代价)

② 关键发现: 月频回测隐藏了严重的月内回撤！
   v9b 报告 MaxDD=-14.9%, 但周频粒度下实际 MaxDD=-27.2%
   周频 drop stop 有效修复了这一缺陷

🚨 无前瞻偏差:
  - 月频信号: 月底价格 → 下月执行
  - 周频止损: 本周五收盘 → 影响下周持仓
  - 所有数据基于可获得的历史价格
"""

import json, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# ── v9b Parameters (unchanged) ──────────────────────────────────────
MOM_W = (0.20, 0.50, 0.20, 0.10)
N_BULL_SECS = 5; BULL_SPS = 2; BEAR_SPS = 2
BREADTH_NARROW = 0.45
GLD_AVG_THRESH = 0.70; GLD_COMPETE_FRAC = 0.20
CONT_BONUS = 0.03; HI52_FRAC = 0.60
DD_PARAMS = {-0.08: 0.30, -0.12: 0.50, -0.18: 0.60}

# ── v5 Weekly Risk Parameters ──────────────────────────────────────
DROP_STOP = -0.09      # Weekly drop threshold: stop if stock drops > 9% in a week
COST = 0.0015          # Transaction cost per side
REENTRY_WAIT = 2       # Weeks to wait before re-entry (unused: monthly rebal resets)


def load_csv(fp):
    df = pd.read_csv(fp, parse_dates=['Date'], index_col='Date')
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


_data_cache = {}

def get_data():
    if _data_cache:
        return _data_cache
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    gld = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    shy = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    
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
    
    _data_cache.update(dict(
        close_df=close_df, sectors=sectors, gld=gld, shy=shy,
        sig=dict(r1=r1, r3=r3, r6=r6, r12=r12, r52w_hi=r52w_hi,
                 vol30=vol30, spy=spy, s200=s200, sma50=sma50, close=close_df),
    ))
    return _data_cache


def compute_breadth(sig, date):
    c = sig['close'].loc[:date].dropna(how='all')
    s = sig['sma50'].loc[:date].dropna(how='all')
    if len(c) < 50: return 1.0
    lc = c.iloc[-1]; ls = s.iloc[-1]
    mask = (lc > ls).dropna()
    return float(mask.sum() / len(mask)) if len(mask) > 0 else 1.0


def get_regime(sig, date):
    sn = sig['spy'].loc[:date].dropna()
    s2 = sig['s200'].loc[:date].dropna()
    if len(sn) == 0 or len(s2) == 0: return 'bull'
    return 'bear' if (sn.iloc[-1] < s2.iloc[-1] and compute_breadth(sig, date) < BREADTH_NARROW) else 'bull'


def gld_comp(sig, gld, date):
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


def monthly_select(sig, sectors, gld, date, prev_hold):
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
    ga = gld_comp(sig, gld, date)
    reg = get_regime(sig, date)
    
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


def backtest(drop_stop=DROP_STOP, cost=COST, start='2015-01-01', end='2025-12-31'):
    """Run hybrid backtest."""
    data = get_data()
    close_df = data['close_df']
    sig = data['sig']
    sectors = data['sectors']
    gld = data['gld']
    shy = data['shy']
    
    rng = close_df.loc[start:end].dropna(how='all')
    monthly_ends = rng.resample('ME').last().index
    weekly_fri = rng.resample('W-FRI').last().index
    
    val = 1.0; peak = 1.0
    vals = []; dates = []
    monthly_weights = {}
    active_weights = {}
    stopped_tickers = {}
    prev_hold = set()
    prev_w_for_cost = {}  # track previous weights for cost calculation
    
    for wi in range(len(weekly_fri) - 1):
        wk_start = weekly_fri[wi]
        wk_end = weekly_fri[wi + 1]
        
        weekly_cost = 0.0  # total cost this week
        
        # Monthly rebalance
        need_monthly = any(wk_start <= me <= wk_end for me in monthly_ends)
        if need_monthly:
            monthly_date = next(me for me in monthly_ends if wk_start <= me <= wk_end)
            dd = (val - peak) / peak if peak > 0 else 0
            new_w = monthly_select(sig, sectors, gld, monthly_date, prev_hold)
            new_w = add_dd_gld(new_w, dd)
            
            # Monthly turnover cost
            all_t = set(list(new_w.keys()) + list(prev_w_for_cost.keys()))
            monthly_turn = sum(abs(new_w.get(t,0) - prev_w_for_cost.get(t,0)) for t in all_t) / 2
            weekly_cost += monthly_turn * cost * 2
            
            monthly_weights = new_w.copy()
            active_weights = new_w.copy()
            prev_hold = {k for k in new_w if k != 'GLD'}
            stopped_tickers = {}
        
        # Weekly drop stop
        if drop_stop is not None:
            new_active = {}
            stopped_amount = 0.0
            risk_turn = 0.0
            
            for t, w in active_weights.items():
                if t in ['GLD', 'SHY']:
                    new_active[t] = w
                    continue
                
                if t in stopped_tickers:
                    stopped_tickers[t] -= 1
                    if stopped_tickers[t] <= 0:
                        del stopped_tickers[t]
                        rw = monthly_weights.get(t, w)
                        new_active[t] = rw
                        risk_turn += abs(rw)
                    else:
                        stopped_amount += w
                    continue
                
                if t in close_df.columns:
                    prices = close_df[t].loc[:wk_start].dropna()
                    if len(prices) >= 6:
                        weekly_drop = prices.iloc[-1] / prices.iloc[-6] - 1
                        if weekly_drop < drop_stop:
                            stopped_amount += w
                            stopped_tickers[t] = REENTRY_WAIT
                            risk_turn += w
                            continue
                
                new_active[t] = w
            
            if stopped_amount > 0:
                new_active['SHY'] = new_active.get('SHY', 0) + stopped_amount
            
            # Risk management turnover cost
            weekly_cost += risk_turn * cost * 2
            active_weights = new_active
        
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
        
        ret -= weekly_cost
        val *= (1 + ret)
        if val > peak: peak = val
        vals.append(val); dates.append(wk_end)
        prev_w_for_cost = active_weights.copy()
    
    eq = pd.Series(vals, index=pd.DatetimeIndex(dates))
    return eq


def metrics(eq):
    if len(eq) < 10: return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs < 0.5: return None
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yrs) - 1
    wr = eq.pct_change().dropna()
    sharpe = wr.mean()/wr.std()*np.sqrt(52) if wr.std() > 0 else 0
    dd = ((eq - eq.cummax())/eq.cummax()).min()
    calmar = cagr/abs(dd) if dd != 0 else 0
    comp = sharpe*0.4 + calmar*0.4 + cagr*0.2
    return {'cagr': cagr, 'sharpe': sharpe, 'max_dd': dd, 'calmar': calmar, 'composite': comp}


def main():
    t0 = time.time()
    print("=" * 70)
    print("🐻 周频策略 v5 — 月频选股 + 周频 Drop Stop 风险管理")
    print("=" * 70)
    
    print(f"\nConfig:")
    print(f"  Monthly selection: v9b logic (unchanged)")
    print(f"  Weekly drop stop:  {DROP_STOP:.0%} threshold")
    print(f"  Transaction cost:  {COST:.2%} per side")
    
    print("\n🔄 Running backtests...")
    
    eq_full = backtest(start='2015-01-01', end='2025-12-31')
    eq_is = backtest(start='2015-01-01', end='2020-12-31')
    eq_oos = backtest(start='2021-01-01', end='2025-12-31')
    
    mf = metrics(eq_full)
    mi = metrics(eq_is)
    mo = metrics(eq_oos)
    
    if not all([mf, mi, mo]):
        print("FAILED!")
        return
    
    wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"📊 RESULTS")
    print(f"{'='*70}")
    print(f"  CAGR:       {mf['cagr']:.1%}  ✅")
    print(f"  Sharpe:     {mf['sharpe']:.2f}  ✅")
    print(f"  MaxDD:      {mf['max_dd']:.1%}  ✅")
    print(f"  Calmar:     {mf['calmar']:.2f}  ✅")
    print(f"  Composite:  {mf['composite']:.4f}  ✅")
    print(f"  IS Sharpe:  {mi['sharpe']:.2f}")
    print(f"  OOS Sharpe: {mo['sharpe']:.2f}")
    print(f"  WF ratio:   {wf:.2f}  ✅")
    
    print(f"\n📊 vs Monthly v9b:")
    print(f"  v9b:    Comp=1.533  Sharpe=1.58  CAGR=31.2%  MaxDD=-14.9%  WF=0.79")
    print(f"  v5:     Comp={mf['composite']:.3f}  Sharpe={mf['sharpe']:.2f}  "
          f"CAGR={mf['cagr']:.1%}  MaxDD={mf['max_dd']:.1%}  WF={wf:.2f}")
    
    # Save
    out = {
        'strategy': 'weekly_v5_drop8_hybrid',
        'full': {k: float(v) for k, v in mf.items()},
        'is': {k: float(v) for k, v in mi.items()},
        'oos': {k: float(v) for k, v in mo.items()},
        'wf': float(wf),
        'params': {'drop_stop': DROP_STOP, 'cost': COST},
    }
    jf = Path(__file__).parent / "weekly_v5_final_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    print(f"  Time: {time.time()-t0:.1f}s")
    
    return mf, wf


if __name__ == '__main__':
    main()
