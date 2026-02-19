"""
stable_v2.py — Adaptive Dual Momentum + Volatility Targeting
代码熊 🐻 | 2026-02-19

核心思路:
1. Dual Momentum: 相对动量选最强资产，绝对动量过滤（负动量→现金）
2. Volatility targeting: 将组合波动率锚定在目标水平（10%年化）
3. Drawdown control: 组合回撤超阈值时降杠杆
4. 月度再平衡

资产池: QQQ, TLT, GLD
基准: 60/40 QQQ/TLT
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"

COMMISSION = 0.0005
SLIPPAGE = 0.001
RISK_FREE = 0.04
INITIAL_CAPITAL = 10000
CASH_YIELD_DAILY = 0.04 / 252

# --- Strategy Parameters ---
MOM_LOOKBACK = 126        # 6-month momentum for relative ranking
ABS_MOM_LOOKBACK = 252    # 12-month for absolute momentum filter
SMA_TREND = 200           # SMA trend filter

VOL_TARGET = 0.10         # 10% annual vol target
VOL_LOOKBACK = 42         # 2-month realized vol
VOL_SCALE_MAX = 1.5       # Max leverage from vol targeting
VOL_SCALE_MIN = 0.3       # Min exposure

DD_THRESHOLD = -0.10      # Start reducing at -10% drawdown
DD_KILL = -0.20            # Maximum cut at -20% drawdown


def load_data():
    assets = {}
    for ticker in ['QQQ', 'TLT', 'GLD']:
        df = pd.read_csv(DATA_DIR / f"{ticker}.csv", parse_dates=['Date'])
        df = df.set_index('Date').sort_index()
        assets[ticker] = df['Close']
    prices = pd.DataFrame(assets).dropna()
    returns = prices.pct_change().dropna()
    return prices, returns


def backtest(prices, returns, start_idx=252):
    tickers = ['QQQ', 'TLT', 'GLD']
    dates = prices.index[start_idx:]
    
    portfolio_values = []
    peak = INITIAL_CAPITAL
    capital = INITIAL_CAPITAL
    holdings = {t: 0.0 for t in tickers}
    holdings['CASH'] = INITIAL_CAPITAL
    last_month = None
    weight_history = []
    trade_costs = 0
    
    for i, date in enumerate(dates):
        idx = start_idx + i
        
        # Apply daily returns
        if i > 0:
            for t in tickers:
                r = returns[t].iloc[idx - 1] if idx - 1 < len(returns) else 0
                holdings[t] *= (1 + r)
            holdings['CASH'] *= (1 + CASH_YIELD_DAILY)
        
        total = sum(holdings.values())
        
        # Track peak for drawdown
        if total > peak:
            peak = total
        
        # Monthly rebalance
        month = (date.year, date.month)
        if month != last_month:
            last_month = month
            
            if idx < ABS_MOM_LOOKBACK:
                # Not enough data, equal weight
                target = {t: 1/3 for t in tickers}
                target['CASH'] = 0.0
            else:
                # Step 1: Compute 6M relative momentum
                mom6 = {}
                for t in tickers:
                    p = prices[t].iloc[idx]
                    p_prev = prices[t].iloc[idx - MOM_LOOKBACK]
                    mom6[t] = p / p_prev - 1
                
                # Step 2: Absolute momentum (12M)
                abs_mom = {}
                for t in tickers:
                    p = prices[t].iloc[idx]
                    p_prev = prices[t].iloc[idx - ABS_MOM_LOOKBACK]
                    abs_mom[t] = p / p_prev - 1
                
                # Step 3: SMA200 trend
                above_sma = {}
                for t in tickers:
                    sma = prices[t].iloc[idx - SMA_TREND:idx].mean()
                    above_sma[t] = prices[t].iloc[idx] > sma
                
                # Step 4: Score = momentum * trend_bonus
                scores = {}
                for t in tickers:
                    if abs_mom[t] < 0 and not above_sma[t]:
                        scores[t] = 0  # Negative absolute momentum + below SMA → skip
                    else:
                        trend_bonus = 1.2 if above_sma[t] else 0.7
                        scores[t] = max(mom6[t], 0) * trend_bonus + (0.05 if above_sma[t] else 0)
                
                total_score = sum(scores.values())
                
                if total_score <= 0:
                    # All assets look bad → 100% cash
                    target = {t: 0.0 for t in tickers}
                    target['CASH'] = 1.0
                else:
                    # Allocate proportional to score
                    target = {t: scores[t] / total_score for t in tickers}
                    target['CASH'] = 0.0
                
                # Step 5: Volatility targeting
                port_returns = returns.iloc[max(0, idx - VOL_LOOKBACK):idx]
                # Estimate portfolio vol with current weights
                w = np.array([target[t] for t in tickers])
                asset_vols = port_returns[tickers].std() * np.sqrt(252)
                # Simple: weighted average vol (conservative estimate)
                port_vol = np.dot(w, asset_vols.values)
                
                if port_vol > 0.01:
                    vol_scale = np.clip(VOL_TARGET / port_vol, VOL_SCALE_MIN, VOL_SCALE_MAX)
                else:
                    vol_scale = 1.0
                
                for t in tickers:
                    target[t] *= vol_scale
                risky_sum = sum(target[t] for t in tickers)
                target['CASH'] = max(0, 1.0 - risky_sum)
                # If leveraged, normalize (no leverage)
                if risky_sum > 1.0:
                    for t in tickers:
                        target[t] /= risky_sum
                    target['CASH'] = 0.0
                
                # Step 6: Drawdown control
                dd = (total - peak) / peak
                if dd < DD_THRESHOLD:
                    # Linear scale down from DD_THRESHOLD to DD_KILL
                    dd_scale = max(0.2, 1.0 - (DD_THRESHOLD - dd) / (DD_THRESHOLD - DD_KILL))
                    for t in tickers:
                        freed = target[t] * (1 - dd_scale)
                        target[t] *= dd_scale
                        target['CASH'] = target.get('CASH', 0) + freed
            
            # Execute rebalance with costs
            for t in tickers:
                current_w = holdings[t] / total if total > 0 else 0
                turnover = abs(target[t] - current_w)
                cost = turnover * total * (COMMISSION + SLIPPAGE)
                trade_costs += cost
                total -= cost
            
            for t in tickers:
                holdings[t] = total * target[t]
            holdings['CASH'] = total * target.get('CASH', 0)
            
            weight_history.append({'Date': date, **{t: target[t] for t in tickers}, 'CASH': target.get('CASH', 0)})
        
        portfolio_values.append(sum(holdings.values()))
    
    pv = pd.Series(portfolio_values, index=dates)
    return pv, weight_history, trade_costs


def metrics(pv, label=""):
    rets = pv.pct_change().dropna()
    years = len(rets) / 252
    cagr = (pv.iloc[-1] / pv.iloc[0]) ** (1/years) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = (cagr - RISK_FREE) / vol if vol > 0 else 0
    dd = (pv / pv.cummax() - 1).min()
    calmar = cagr / abs(dd) if dd != 0 else 0
    monthly = pv.resample('ME').last().pct_change().dropna()
    wr = (monthly > 0).mean()
    return {
        'label': label, 'final': pv.iloc[-1], 'cagr': cagr, 'vol': vol,
        'sharpe': sharpe, 'maxdd': dd, 'calmar': calmar, 'winrate': wr
    }


def fmt(m):
    return (f"  {m['label']:28s} | ${m['final']:>10,.0f} | CAGR {m['cagr']:6.1%} | "
            f"Vol {m['vol']:5.1%} | MaxDD {m['maxdd']:6.1%} | Sharpe {m['sharpe']:5.2f} | "
            f"Calmar {m['calmar']:5.2f} | WR {m['winrate']:5.1%}")


def buy_hold_6040(prices, start_idx=252):
    dates = prices.index[start_idx:]
    rets = prices[['QQQ','TLT']].pct_change().iloc[start_idx:]
    pr = rets['QQQ']*0.6 + rets['TLT']*0.4
    return INITIAL_CAPITAL * (1 + pr).cumprod()


def main():
    prices, returns = load_data()
    total = len(prices)
    split = int(total * 0.6)
    
    print(f"Data: {prices.index[0].date()} → {prices.index[-1].date()} ({total} days)")
    print(f"Train: → {prices.index[split-1].date()} | Test: {prices.index[split].date()} →")
    
    pv, wh, costs = backtest(prices, returns, 252)
    bh = buy_hold_6040(prices, 252)
    
    print("\n=== FULL PERIOD ===")
    m_full = metrics(pv, "Stable v2 (Full)")
    m_bh = metrics(bh, "60/40 B&H (Full)")
    print(fmt(m_full))
    print(fmt(m_bh))
    print(f"  Costs: ${costs:,.0f}")
    
    # Train
    train_end = prices.index[split - 1]
    pv_tr = pv.loc[:train_end]
    bh_tr = bh.loc[:train_end]
    print("\n=== TRAIN (in-sample) ===")
    m_tr = metrics(pv_tr, "Stable v2 (Train)")
    print(fmt(m_tr))
    print(fmt(metrics(bh_tr, "60/40 (Train)")))
    
    # Test
    pv_te = pv.loc[prices.index[split]:]
    pv_te = pv_te / pv_te.iloc[0] * INITIAL_CAPITAL
    bh_te = bh.loc[prices.index[split]:]
    bh_te = bh_te / bh_te.iloc[0] * INITIAL_CAPITAL
    print("\n=== TEST (out-of-sample) ===")
    m_te = metrics(pv_te, "Stable v2 (Test)")
    print(fmt(m_te))
    print(fmt(metrics(bh_te, "60/40 (Test)")))
    
    # Overfit check
    deg = (m_tr['sharpe'] - m_te['sharpe']) / abs(m_tr['sharpe']) * 100 if m_tr['sharpe'] != 0 else 0
    print(f"\nOverfit: Train Sharpe {m_tr['sharpe']:.2f} → Test {m_te['sharpe']:.2f} | Δ {deg:.0f}% {'✅' if abs(deg)<30 else '⚠️'}")
    
    score = m_full['sharpe']*0.4 + m_full['calmar']*0.4 + m_full['cagr']*0.2
    print(f"Composite: {score:.3f}")
    
    # Weight stats
    wdf = pd.DataFrame(wh)
    print("\nAvg weights:")
    for c in ['QQQ','TLT','GLD','CASH']:
        print(f"  {c}: {wdf[c].mean():.1%}")


if __name__ == "__main__":
    main()
