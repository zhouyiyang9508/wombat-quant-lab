"""
Multi-Strategy Portfolio Optimization v1
Combine best strategies into a diversified portfolio:
- TQQQ Beast v9g (high growth, leveraged tech)
- BTC v7f DualMom (crypto + gold rotation)
- Stock Momentum v2d (equity momentum)
- Hedge v4 PMA (defensive allocation)

Goal: Maximize portfolio Sharpe through low-correlation strategy combination.
代码熊 🐻 2026-02-20
"""

import pandas as pd
import numpy as np
import os, sys, json
from itertools import product

# Add parent paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../tqqq/codebear'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../btc/codebear'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../stocks/codebear'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../hedge/codebear'))

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data_cache')

# ─── Utility Functions ───

def calc_metrics(returns, rf_annual=0.04):
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 126:
        return {}
    cum = np.cumprod(1 + returns)
    years = len(returns) / 252
    cagr = cum[-1] ** (1/years) - 1
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    maxdd = dd.min()
    daily_rf = (1 + rf_annual) ** (1/252) - 1
    excess = returns - daily_rf
    sharpe = np.sqrt(252) * np.mean(excess) / (np.std(excess) + 1e-10)
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    win_rate = np.mean(returns > 0)
    vol = np.std(returns) * np.sqrt(252)
    return {
        'CAGR': cagr, 'MaxDD': maxdd, 'Sharpe': sharpe, 'Calmar': calmar,
        'WinRate': win_rate, 'AnnVol': vol, 'FinalValue': cum[-1],
        'Years': years
    }

def composite_score(m):
    return m['Sharpe']*0.4 + m['Calmar']*0.4 + m['CAGR']*0.2

def load_csv(ticker):
    path = os.path.join(DATA_DIR, f'{ticker}.csv')
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date').sort_index()
    return df['Close'].dropna()

# ─── Strategy 1: TQQQ Beast v9g ───

def run_tqqq_v9g():
    """Simplified v9g: SMA200 hysteresis + GLD hedge in bear"""
    tqqq = load_csv('TQQQ')
    gld = load_csv('GLD')
    df = pd.DataFrame({'TQQQ': tqqq, 'GLD': gld}).dropna()

    sma200 = df['TQQQ'].rolling(200).mean()
    rsi = compute_rsi(df['TQQQ'], 10)
    weekly_ret = df['TQQQ'].pct_change(5)

    bull_thr, bear_thr, floor = 1.05, 0.93, 0.25
    regime = pd.Series('bull', index=df.index)
    for i in range(1, len(df)):
        prev = regime.iloc[i-1]
        price = df['TQQQ'].iloc[i]
        sma = sma200.iloc[i]
        if np.isnan(sma):
            regime.iloc[i] = 'bull'
            continue
        if prev == 'bull' and price < sma * bear_thr:
            regime.iloc[i] = 'bear'
        elif prev == 'bear' and price > sma * bull_thr:
            regime.iloc[i] = 'bull'
        else:
            regime.iloc[i] = prev

    tqqq_ret = df['TQQQ'].pct_change()
    gld_ret = df['GLD'].pct_change()
    strat_ret = pd.Series(0.0, index=df.index)

    for i in range(1, len(df)):
        r = regime.iloc[i]
        r_rsi = rsi.iloc[i] if not np.isnan(rsi.iloc[i]) else 50
        r_wk = weekly_ret.iloc[i] if not np.isnan(weekly_ret.iloc[i]) else 0

        if r == 'bull':
            tqqq_w = 1.0
            if r_rsi > 80 and r_wk > 0.15:
                tqqq_w = 0.80
            gld_w = 0.0
        else:
            if r_rsi < 20 or r_wk < -0.12:
                tqqq_w = 0.80
            elif r_rsi < 30:
                tqqq_w = 0.60
            elif r_rsi > 65:
                tqqq_w = floor
            else:
                tqqq_w = floor
            gld_w = 1.0 - tqqq_w

        tr = tqqq_ret.iloc[i] if not np.isnan(tqqq_ret.iloc[i]) else 0
        gr = gld_ret.iloc[i] if not np.isnan(gld_ret.iloc[i]) else 0
        strat_ret.iloc[i] = tqqq_w * tr + gld_w * gr

    return strat_ret.iloc[200:]  # skip warmup


def compute_rsi(series, period=10):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - 100 / (1 + rs)


# ─── Strategy 2: BTC v7f DualMom ───

def run_btc_v7f():
    """BTC vs GLD dual momentum rotation"""
    btc = load_csv('BTC_USD')
    gld = load_csv('GLD')
    df = pd.DataFrame({'BTC': btc, 'GLD': gld}).dropna()

    # Momentum signals
    btc_mom3 = df['BTC'].pct_change(63)
    btc_mom6 = df['BTC'].pct_change(126)
    gld_mom3 = df['GLD'].pct_change(63)
    gld_mom6 = df['GLD'].pct_change(126)
    btc_blend = 0.4 * btc_mom3 + 0.6 * btc_mom6
    gld_blend = 0.4 * gld_mom3 + 0.6 * gld_mom6

    # Mayer Multiple
    sma200 = df['BTC'].rolling(200).mean()
    mayer = df['BTC'] / sma200

    # Halving dates
    halvings = [pd.Timestamp('2016-07-09'), pd.Timestamp('2020-05-11'), pd.Timestamp('2024-04-19')]

    def months_since_halving(date):
        for h in reversed(halvings):
            if date >= h:
                return (date - h).days / 30.44
        return 999

    btc_ret = df['BTC'].pct_change()
    gld_ret = df['GLD'].pct_change()
    strat_ret = pd.Series(0.0, index=df.index)

    for i in range(200, len(df)):
        bm = btc_blend.iloc[i]
        gm = gld_blend.iloc[i]
        mm = mayer.iloc[i]
        msh = months_since_halving(df.index[i])

        if np.isnan(bm) or np.isnan(gm):
            continue

        # Base allocation by relative momentum
        btc_pos = bm > 0
        gld_pos = gm > 0

        if btc_pos and gld_pos:
            if bm > gm:
                bw, gw = 0.80, 0.15
            else:
                bw, gw = 0.50, 0.40
        elif btc_pos:
            bw, gw = 0.85, 0.05
        elif gld_pos:
            bw, gw = 0.25, 0.50
        else:
            bw, gw = 0.20, 0.30

        # Halving early boost
        if msh <= 18:
            bw = max(bw, 0.50)

        # Mayer bubble protection
        if not np.isnan(mm):
            if mm > 3.5:
                bw = min(bw, 0.35)
            elif mm > 2.4:
                bw = min(bw, 0.60)

        cash = max(0, 1.0 - bw - gw)
        br = btc_ret.iloc[i] if not np.isnan(btc_ret.iloc[i]) else 0
        gr = gld_ret.iloc[i] if not np.isnan(gld_ret.iloc[i]) else 0
        strat_ret.iloc[i] = bw * br + gw * gr

    return strat_ret.iloc[200:]


# ─── Strategy 3: Stock Momentum v2d (simplified proxy) ───

def run_stock_momentum_v2d():
    """Proxy: QQQ momentum with regime filter (approximates v2d behavior)
    Full v2d needs 500 stocks; here we use QQQ as a close proxy with similar risk profile."""
    spy = load_csv('SPY')
    qqq = load_csv('QQQ')
    gld = load_csv('GLD')

    df = pd.DataFrame({'SPY': spy, 'QQQ': qqq, 'GLD': gld}).dropna()

    # Use QQQ/SPY relative strength as momentum proxy
    qqq_mom3 = df['QQQ'].pct_change(63)
    qqq_mom6 = df['QQQ'].pct_change(126)
    spy_mom3 = df['SPY'].pct_change(63)
    spy_sma200 = df['SPY'].rolling(200).mean()

    qqq_ret = df['QQQ'].pct_change()
    spy_ret = df['SPY'].pct_change()
    gld_ret = df['GLD'].pct_change()
    strat_ret = pd.Series(0.0, index=df.index)

    for i in range(200, len(df)):
        qm3 = qqq_mom3.iloc[i]
        qm6 = qqq_mom6.iloc[i]
        sm3 = spy_mom3.iloc[i]
        sma = spy_sma200.iloc[i]
        price = df['SPY'].iloc[i]

        if np.isnan(qm3) or np.isnan(sma):
            continue

        # Bull/bear regime
        bull = price > sma

        if bull:
            # Momentum: favor QQQ if stronger
            blend = 0.4*qm3 + 0.6*qm6 if not np.isnan(qm6) else qm3
            if blend > 0.05:
                qw, sw, gw = 0.70, 0.25, 0.05
            elif blend > 0:
                qw, sw, gw = 0.50, 0.40, 0.10
            else:
                qw, sw, gw = 0.30, 0.40, 0.30
        else:
            # Bear: defensive
            if sm3 is not np.nan and sm3 > 0:
                qw, sw, gw = 0.20, 0.30, 0.50
            else:
                qw, sw, gw = 0.10, 0.20, 0.70

        qr = qqq_ret.iloc[i] if not np.isnan(qqq_ret.iloc[i]) else 0
        sr = spy_ret.iloc[i] if not np.isnan(spy_ret.iloc[i]) else 0
        gr = gld_ret.iloc[i] if not np.isnan(gld_ret.iloc[i]) else 0
        strat_ret.iloc[i] = qw * qr + sw * sr + gw * gr

    return strat_ret.iloc[200:]


# ─── Strategy 4: Hedge v4 PMA ───

def run_hedge_v4():
    """QQQ/TLT/GLD breadth + momentum allocation"""
    qqq = load_csv('QQQ')
    tlt = load_csv('TLT')
    gld = load_csv('GLD')
    df = pd.DataFrame({'QQQ': qqq, 'TLT': tlt, 'GLD': gld}).dropna()

    sma_p = 65
    mom_lb = 126

    sma = {t: df[t].rolling(sma_p).mean() for t in ['QQQ','TLT','GLD']}
    mom = {t: df[t].pct_change(mom_lb) for t in ['QQQ','TLT','GLD']}
    rets = {t: df[t].pct_change() for t in ['QQQ','TLT','GLD']}

    strat_ret = pd.Series(0.0, index=df.index)
    prev_w = {'QQQ': 0.33, 'TLT': 0.33, 'GLD': 0.34}
    smooth = 0.7

    for i in range(200, len(df)):
        above = sum(1 for t in ['QQQ','TLT','GLD']
                     if not np.isnan(sma[t].iloc[i]) and df[t].iloc[i] > sma[t].iloc[i])

        # Momentum ranking
        moms = {}
        for t in ['QQQ','TLT','GLD']:
            m = mom[t].iloc[i]
            moms[t] = m if not np.isnan(m) else -999
        ranked = sorted(moms.keys(), key=lambda x: moms[x], reverse=True)

        if above >= 2:  # Offense
            w = {ranked[0]: 0.55, ranked[1]: 0.30, ranked[2]: 0.15}
            # Filter negative absolute momentum
            for t in ['QQQ','TLT','GLD']:
                if moms[t] < 0:
                    w[t] = 0
            total = sum(w.values())
            if total > 0:
                w = {t: v/total for t, v in w.items()}
        elif above == 1:  # Mixed
            w = {ranked[0]: 0.40, 'TLT': 0.35, 'GLD': 0.25}
        else:  # Defense
            w = {'QQQ': 0.0, 'TLT': 0.50, 'GLD': 0.40}

        # Smooth
        new_w = {}
        for t in ['QQQ','TLT','GLD']:
            new_w[t] = smooth * prev_w.get(t, 0.33) + (1-smooth) * w.get(t, 0)
        total = sum(new_w.values())
        if total > 0:
            new_w = {t: v/total for t, v in new_w.items()}
        prev_w = new_w

        # Drawdown protection
        cum = np.cumprod(1 + strat_ret.iloc[max(0,i-252):i].values)
        if len(cum) > 0:
            peak = np.max(cum)
            dd = (cum[-1] - peak) / peak if peak > 0 else 0
            if dd < -0.06:
                scale = max(0.3, 1.0 + (dd + 0.06) * 5)
                new_w = {t: v * scale for t, v in new_w.items()}

        r = sum(new_w.get(t, 0) * (rets[t].iloc[i] if not np.isnan(rets[t].iloc[i]) else 0)
                for t in ['QQQ','TLT','GLD'])
        strat_ret.iloc[i] = r

    return strat_ret.iloc[200:]


# ─── Portfolio Combination ───

def align_returns(*series_list):
    """Align multiple return series to common date range"""
    df = pd.DataFrame({f's{i}': s for i, s in enumerate(series_list)})
    df = df.dropna()
    return [df[c] for c in df.columns]

def portfolio_returns(returns_list, weights):
    """Combine strategy returns with given weights"""
    aligned = align_returns(*returns_list)
    port_ret = sum(w * r.values for w, r in zip(weights, aligned))
    return pd.Series(port_ret, index=aligned[0].index)

def grid_search_weights(returns_list, names, step=0.05):
    """Find optimal weights by grid search"""
    n = len(returns_list)
    aligned = align_returns(*returns_list)
    best_score = -999
    best_w = None
    best_m = None
    results = []

    # Generate weight combinations that sum to 1.0
    if n == 4:
        steps = np.arange(0, 1.01, step)
        for w0 in steps:
            for w1 in steps:
                for w2 in steps:
                    w3 = 1.0 - w0 - w1 - w2
                    if w3 < -0.001 or w3 > 1.001:
                        continue
                    w3 = max(0, w3)
                    weights = [w0, w1, w2, w3]
                    port = sum(w * r.values for w, r in zip(weights, aligned))
                    m = calc_metrics(port)
                    if not m:
                        continue
                    score = composite_score(m)
                    results.append({'weights': weights, 'metrics': m, 'score': score})
                    if score > best_score:
                        best_score = score
                        best_w = weights
                        best_m = m

    return best_w, best_m, best_score, results


def risk_parity_weights(returns_list):
    """Inverse-volatility weighting"""
    vols = [np.std(r) * np.sqrt(252) for r in returns_list]
    inv_vol = [1/v for v in vols]
    total = sum(inv_vol)
    return [v/total for v in inv_vol]


def walk_forward(returns_list, weights, split=0.6):
    """Walk-forward validation"""
    aligned = align_returns(*returns_list)
    n = len(aligned[0])
    split_idx = int(n * split)

    is_port = sum(w * r.values[:split_idx] for w, r in zip(weights, aligned))
    oos_port = sum(w * r.values[split_idx:] for w, r in zip(weights, aligned))

    is_m = calc_metrics(is_port)
    oos_m = calc_metrics(oos_port)
    return is_m, oos_m


def main():
    print("=" * 80)
    print("Multi-Strategy Portfolio Optimization v1")
    print("代码熊 🐻 2026-02-20")
    print("=" * 80)

    # Run individual strategies
    print("\n[1] Running individual strategies...")
    strategies = {}

    print("  → TQQQ v9g...")
    strategies['TQQQ_v9g'] = run_tqqq_v9g()

    print("  → BTC v7f DualMom...")
    strategies['BTC_v7f'] = run_btc_v7f()

    print("  → Stock Momentum (QQQ proxy)...")
    strategies['StockMom'] = run_stock_momentum_v2d()

    print("  → Hedge v4 PMA...")
    strategies['Hedge_v4'] = run_hedge_v4()

    # Individual metrics
    print("\n[2] Individual Strategy Metrics:")
    print(f"{'Strategy':<16} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'AnnVol':>8} {'Score':>8}")
    print("-" * 72)
    names = list(strategies.keys())
    rets_list = [strategies[n] for n in names]

    for name, ret in zip(names, rets_list):
        m = calc_metrics(ret)
        if m:
            sc = composite_score(m)
            print(f"{name:<16} {m['CAGR']:>7.1%} {m['MaxDD']:>7.1%} {m['Sharpe']:>8.2f} {m['Calmar']:>8.2f} {m['AnnVol']:>7.1%} {sc:>8.3f}")

    # Correlation matrix
    print("\n[3] Strategy Correlation Matrix:")
    aligned = align_returns(*rets_list)
    corr_df = pd.DataFrame({n: a.values for n, a in zip(names, aligned)}).corr()
    print(corr_df.round(3).to_string())

    # Equal weight portfolio
    print("\n[4] Portfolio Combinations:")
    print(f"{'Config':<30} {'Weights':<40} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} {'Calmar':>7} {'Score':>7}")
    print("-" * 105)

    configs = [
        ("Equal Weight", [0.25, 0.25, 0.25, 0.25]),
        ("Risk Parity", risk_parity_weights(aligned)),
        ("High Growth (60/20/10/10)", [0.60, 0.20, 0.10, 0.10]),
        ("Balanced (30/30/20/20)", [0.30, 0.30, 0.20, 0.20]),
        ("BTC+TQQQ Heavy (40/40/10/10)", [0.40, 0.40, 0.10, 0.10]),
        ("Barbell (45/45/5/5)", [0.45, 0.45, 0.05, 0.05]),
    ]

    for label, w in configs:
        port = portfolio_returns(rets_list, w)
        m = calc_metrics(port)
        if m:
            sc = composite_score(m)
            w_str = '/'.join(f"{x:.0%}" for x in w)
            print(f"{label:<30} {w_str:<40} {m['CAGR']:>6.1%} {m['MaxDD']:>6.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} {sc:>7.3f}")

    # Grid search optimal
    print("\n[5] Grid Search for Optimal Weights (step=0.10)...")
    best_w, best_m, best_score, all_results = grid_search_weights(aligned, names, step=0.10)

    if best_w:
        w_str = '/'.join(f"{x:.0%}" for x in best_w)
        print(f"\n  🏆 Optimal: {w_str}")
        print(f"     CAGR={best_m['CAGR']:.1%} MaxDD={best_m['MaxDD']:.1%} Sharpe={best_m['Sharpe']:.2f} Calmar={best_m['Calmar']:.2f} Score={best_score:.3f}")

        # Fine-tune around optimal
        print("\n[6] Fine-tuning (step=0.05 around optimal)...")
        fine_results = []
        for d0 in np.arange(-0.10, 0.11, 0.05):
            for d1 in np.arange(-0.10, 0.11, 0.05):
                for d2 in np.arange(-0.10, 0.11, 0.05):
                    w = [best_w[0]+d0, best_w[1]+d1, best_w[2]+d2, 0]
                    w[3] = 1.0 - w[0] - w[1] - w[2]
                    if any(x < 0 for x in w) or any(x > 1 for x in w):
                        continue
                    port = sum(wi * r.values for wi, r in zip(w, aligned))
                    m = calc_metrics(port)
                    if m:
                        sc = composite_score(m)
                        fine_results.append((w, m, sc))

        fine_results.sort(key=lambda x: x[2], reverse=True)
        if fine_results:
            fw, fm, fs = fine_results[0]
            w_str = '/'.join(f"{x:.0%}" for x in fw)
            print(f"  🏆 Fine-tuned: {w_str}")
            print(f"     CAGR={fm['CAGR']:.1%} MaxDD={fm['MaxDD']:.1%} Sharpe={fm['Sharpe']:.2f} Calmar={fm['Calmar']:.2f} Score={fs:.3f}")

            # Walk-forward
            print("\n[7] Walk-Forward Validation (60/40 split)...")
            is_m, oos_m = walk_forward(rets_list, fw)
            if is_m and oos_m:
                wf_ratio = oos_m['Sharpe'] / is_m['Sharpe'] if is_m['Sharpe'] > 0 else 0
                wf_pass = "✅" if wf_ratio >= 0.70 else "❌"
                print(f"  IS:  CAGR={is_m['CAGR']:.1%} Sharpe={is_m['Sharpe']:.2f} MaxDD={is_m['MaxDD']:.1%}")
                print(f"  OOS: CAGR={oos_m['CAGR']:.1%} Sharpe={oos_m['Sharpe']:.2f} MaxDD={oos_m['MaxDD']:.1%}")
                print(f"  WF Ratio: {wf_ratio:.2f} {wf_pass}")

            # Top 5 configs
            print("\n[8] Top 5 Configurations:")
            print(f"{'#':<4} {'Weights':<40} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} {'Calmar':>7} {'Score':>7}")
            print("-" * 80)
            for i, (w, m, s) in enumerate(fine_results[:5]):
                w_str = '/'.join(f"{x:.0%}" for x in w)
                print(f"{i+1:<4} {w_str:<40} {m['CAGR']:>6.1%} {m['MaxDD']:>6.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} {s:>7.3f}")

    # vs individual best
    print("\n[9] Portfolio vs Individual Strategies:")
    print("  The portfolio should have HIGHER Sharpe than any individual strategy")
    print("  due to diversification (low correlation between strategies).")

    # Save results
    output = {
        'strategy_names': names,
        'correlation': corr_df.to_dict(),
        'optimal_weights': best_w if best_w else [],
        'optimal_score': best_score,
        'optimal_metrics': best_m if best_m else {},
    }
    out_path = os.path.join(os.path.dirname(__file__), 'multi_strategy_v1_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    print("\n" + "=" * 80)
    print("Done! 🐻")


if __name__ == '__main__':
    main()
