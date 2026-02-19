"""
stable_v1.py — 稳健对冲基金风格动态资产配置
代码熊 🐻 | 2026-02-19

资产池: QQQ (股票增长) + TLT (国债对冲) + GLD (黄金避险)
方法: 双动量 + 风险平价混合
目标: CAGR 15-25%, MaxDD < 20%, Sharpe > 1.5

核心逻辑:
1. 相对动量: 12M/6M/3M/1M 加权动量选择资产倾斜
2. 绝对动量: 资产自身动量为负时减仓，转入现金
3. 风险平价基础权重: 按波动率倒数分配
4. 月度再平衡（低频避免交易成本）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"

# === Parameters ===
COMMISSION = 0.0005  # 0.05% per side
SLIPPAGE = 0.001     # 0.1%
RISK_FREE = 0.04     # 4% annual
INITIAL_CAPITAL = 10000

# Momentum lookback windows and weights
MOM_WINDOWS = [252, 126, 63, 21]  # 12M, 6M, 3M, 1M
MOM_WEIGHTS = [0.3, 0.3, 0.25, 0.15]

# Risk parity vol lookback
VOL_LOOKBACK = 63  # 3 months

# Absolute momentum threshold (annualized)
ABS_MOM_THRESHOLD = 0.0  # positive = in, negative = cash

# Max allocation to single asset
MAX_WEIGHT = 0.60
MIN_WEIGHT = 0.05

# Cash yield (approximate T-bill)
CASH_YIELD_DAILY = 0.04 / 252


def load_data():
    """Load and align QQQ, TLT, GLD data."""
    assets = {}
    for ticker in ['QQQ', 'TLT', 'GLD']:
        df = pd.read_csv(DATA_DIR / f"{ticker}.csv", parse_dates=['Date'])
        df = df.set_index('Date').sort_index()
        assets[ticker] = df['Close']
    
    prices = pd.DataFrame(assets).dropna()
    returns = prices.pct_change().dropna()
    return prices, returns


def compute_momentum(prices, window):
    """Simple total return momentum."""
    return prices / prices.shift(window) - 1


def compute_weights(prices, returns, date_idx, train_only=True):
    """
    Compute target weights for given date index.
    Returns dict of weights for QQQ, TLT, GLD, CASH.
    """
    tickers = ['QQQ', 'TLT', 'GLD']
    
    # Need enough history
    if date_idx < 252:
        # Equal weight as default
        return {'QQQ': 0.33, 'TLT': 0.33, 'GLD': 0.34, 'CASH': 0.0}
    
    hist_prices = prices.iloc[:date_idx + 1]
    hist_returns = returns.iloc[:date_idx]
    
    # --- Step 1: Risk Parity base weights ---
    recent_vol = hist_returns.iloc[-VOL_LOOKBACK:].std() * np.sqrt(252)
    inv_vol = 1.0 / recent_vol.clip(lower=0.01)
    rp_weights = inv_vol / inv_vol.sum()
    
    # --- Step 2: Momentum score ---
    mom_scores = {}
    for ticker in tickers:
        score = 0.0
        for w, wt in zip(MOM_WINDOWS, MOM_WEIGHTS):
            if len(hist_prices) > w:
                m = hist_prices[ticker].iloc[-1] / hist_prices[ticker].iloc[-w] - 1
                score += m * wt
            else:
                score += 0
        mom_scores[ticker] = score
    
    # Normalize momentum scores to [-1, 1] range via rank
    scores = pd.Series(mom_scores)
    ranks = scores.rank()  # 1, 2, 3
    norm_ranks = (ranks - ranks.mean()) / max(ranks.std(), 0.01)  # z-score
    
    # --- Step 3: Combine RP + Momentum tilt ---
    # Tilt = RP_weight * (1 + 0.5 * norm_rank)
    combined = {}
    for ticker in tickers:
        combined[ticker] = rp_weights[ticker] * (1 + 0.5 * norm_ranks[ticker])
    
    # Normalize
    total = sum(combined.values())
    for ticker in tickers:
        combined[ticker] /= total
    
    # --- Step 4: Absolute momentum filter ---
    # If asset's 6M momentum is negative, reduce allocation, move to cash
    cash = 0.0
    for ticker in tickers:
        if len(hist_prices) > 126:
            abs_mom = hist_prices[ticker].iloc[-1] / hist_prices[ticker].iloc[-126] - 1
            ann_mom = abs_mom * 2  # annualize roughly
            if ann_mom < ABS_MOM_THRESHOLD:
                # Scale down proportional to how negative
                scale = max(0.2, 1 + ann_mom)  # at -80% ann mom → 0.2x
                freed = combined[ticker] * (1 - scale)
                combined[ticker] *= scale
                cash += freed
    
    # --- Step 5: Enforce limits ---
    for ticker in tickers:
        combined[ticker] = np.clip(combined[ticker], MIN_WEIGHT, MAX_WEIGHT)
    
    # Re-normalize risky assets
    risky_total = sum(combined[ticker] for ticker in tickers)
    if risky_total > 0:
        target_risky = 1.0 - cash
        for ticker in tickers:
            combined[ticker] = combined[ticker] / risky_total * target_risky
    
    combined['CASH'] = cash
    return combined


def backtest(prices, returns, start_idx=252):
    """Run monthly rebalance backtest."""
    tickers = ['QQQ', 'TLT', 'GLD']
    dates = prices.index[start_idx:]
    
    portfolio_value = [INITIAL_CAPITAL]
    holdings = {'QQQ': 0, 'TLT': 0, 'GLD': 0, 'CASH': INITIAL_CAPITAL}
    current_weights = None
    last_rebal_month = None
    
    weight_history = []
    trade_costs = 0
    n_rebalances = 0
    
    for i, date in enumerate(dates):
        idx = start_idx + i
        
        # Daily returns
        if i > 0:
            for ticker in tickers:
                daily_ret = returns[ticker].iloc[idx - 1] if idx - 1 < len(returns) else 0
                holdings[ticker] *= (1 + daily_ret)
            holdings['CASH'] *= (1 + CASH_YIELD_DAILY)
        
        # Monthly rebalance (first trading day of month)
        current_month = (date.year, date.month)
        if last_rebal_month is None or current_month != last_rebal_month:
            last_rebal_month = current_month
            
            total_value = sum(holdings.values())
            target_weights = compute_weights(prices, returns, idx)
            
            # Calculate turnover and apply costs
            for ticker in tickers:
                current_w = holdings[ticker] / total_value if total_value > 0 else 0
                target_w = target_weights[ticker]
                turnover = abs(target_w - current_w)
                cost = turnover * total_value * (COMMISSION + SLIPPAGE)
                trade_costs += cost
                total_value -= cost
            
            # Rebalance
            for ticker in tickers:
                holdings[ticker] = total_value * target_weights[ticker]
            holdings['CASH'] = total_value * target_weights['CASH']
            
            n_rebalances += 1
            weight_history.append({
                'Date': date,
                **{k: v for k, v in target_weights.items()}
            })
        
        portfolio_value.append(sum(holdings.values()))
    
    # Build result series
    pv = pd.Series(portfolio_value[1:], index=dates)
    return pv, weight_history, trade_costs, n_rebalances


def compute_metrics(pv, label="Strategy"):
    """Compute standard performance metrics."""
    returns = pv.pct_change().dropna()
    days = len(returns)
    years = days / 252
    
    total_return = pv.iloc[-1] / pv.iloc[0] - 1
    cagr = (pv.iloc[-1] / pv.iloc[0]) ** (1 / years) - 1
    
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - RISK_FREE) / ann_vol if ann_vol > 0 else 0
    
    # Max drawdown
    running_max = pv.cummax()
    drawdown = (pv - running_max) / running_max
    max_dd = drawdown.min()
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # Win rate (monthly)
    monthly = pv.resample('ME').last().pct_change().dropna()
    win_rate = (monthly > 0).mean()
    
    return {
        'Label': label,
        'Final Value': f"${pv.iloc[-1]:,.0f}",
        'CAGR': f"{cagr:.1%}",
        'MaxDD': f"{max_dd:.1%}",
        'Ann Vol': f"{ann_vol:.1%}",
        'Sharpe': f"{sharpe:.2f}",
        'Calmar': f"{calmar:.2f}",
        'Win Rate (M)': f"{win_rate:.1%}",
        '_cagr': cagr,
        '_maxdd': max_dd,
        '_sharpe': sharpe,
        '_calmar': calmar,
    }


def buy_and_hold_benchmark(prices, start_idx=252):
    """60/40 QQQ/TLT benchmark."""
    dates = prices.index[start_idx:]
    tickers = ['QQQ', 'TLT']
    weights = [0.6, 0.4]
    
    # Daily returns
    daily_rets = prices[tickers].pct_change().iloc[start_idx:]
    port_ret = (daily_rets * weights).sum(axis=1)
    pv = INITIAL_CAPITAL * (1 + port_ret).cumprod()
    return pv


def run_walkforward():
    """Walk-forward: 60% train / 40% test."""
    prices, returns = load_data()
    
    total_days = len(prices)
    train_end = int(total_days * 0.6)
    test_start_idx = train_end
    
    print(f"Total data: {prices.index[0].date()} → {prices.index[-1].date()} ({total_days} days)")
    print(f"Train: {prices.index[252].date()} → {prices.index[train_end-1].date()}")
    print(f"Test:  {prices.index[test_start_idx].date()} → {prices.index[-1].date()}")
    print()
    
    # --- Full period ---
    print("=" * 70)
    print("FULL PERIOD BACKTEST")
    print("=" * 70)
    pv_full, wh_full, costs_full, n_rebal = backtest(prices, returns, start_idx=252)
    m_full = compute_metrics(pv_full, "Stable v1 (Full)")
    
    bh_full = buy_and_hold_benchmark(prices, start_idx=252)
    m_bh_full = compute_metrics(bh_full, "60/40 B&H (Full)")
    
    for m in [m_full, m_bh_full]:
        print(f"  {m['Label']:25s} | {m['Final Value']:>12s} | CAGR {m['CAGR']:>7s} | MaxDD {m['MaxDD']:>7s} | Sharpe {m['Sharpe']:>5s} | Calmar {m['Calmar']:>5s} | WinRate {m['Win Rate (M)']}")
    print(f"  Trade costs: ${costs_full:,.0f} | Rebalances: {n_rebal}")
    
    # --- Train period ---
    print()
    print("=" * 70)
    print("TRAIN PERIOD (in-sample)")
    print("=" * 70)
    pv_train = pv_full.loc[:prices.index[train_end-1]]
    bh_train = bh_full.loc[:prices.index[train_end-1]]
    m_train = compute_metrics(pv_train, "Stable v1 (Train)")
    m_bh_train = compute_metrics(bh_train, "60/40 B&H (Train)")
    for m in [m_train, m_bh_train]:
        print(f"  {m['Label']:25s} | {m['Final Value']:>12s} | CAGR {m['CAGR']:>7s} | MaxDD {m['MaxDD']:>7s} | Sharpe {m['Sharpe']:>5s} | Calmar {m['Calmar']:>5s}")
    
    # --- Test period ---
    print()
    print("=" * 70)
    print("TEST PERIOD (out-of-sample)")
    print("=" * 70)
    pv_test = pv_full.loc[prices.index[test_start_idx]:]
    # Re-index to start from initial value
    pv_test_norm = pv_test / pv_test.iloc[0] * INITIAL_CAPITAL
    bh_test = bh_full.loc[prices.index[test_start_idx]:]
    bh_test_norm = bh_test / bh_test.iloc[0] * INITIAL_CAPITAL
    
    m_test = compute_metrics(pv_test_norm, "Stable v1 (Test/OOS)")
    m_bh_test = compute_metrics(bh_test_norm, "60/40 B&H (Test/OOS)")
    for m in [m_test, m_bh_test]:
        print(f"  {m['Label']:25s} | {m['Final Value']:>12s} | CAGR {m['CAGR']:>7s} | MaxDD {m['MaxDD']:>7s} | Sharpe {m['Sharpe']:>5s} | Calmar {m['Calmar']:>5s}")
    
    # --- Overfitting check ---
    print()
    print("=" * 70)
    print("OVERFITTING CHECK")
    print("=" * 70)
    train_sharpe = m_train['_sharpe']
    test_sharpe = m_test['_sharpe']
    degradation = (train_sharpe - test_sharpe) / abs(train_sharpe) * 100 if train_sharpe != 0 else 0
    print(f"  Train Sharpe: {train_sharpe:.2f}")
    print(f"  Test Sharpe:  {test_sharpe:.2f}")
    print(f"  Degradation:  {degradation:.1f}% {'✅ PASS (<30%)' if abs(degradation) < 30 else '⚠️ CAUTION'}")
    
    # --- Composite score ---
    print()
    print("=" * 70)
    print("COMPOSITE SCORE (Full Period)")
    print("=" * 70)
    score = m_full['_sharpe'] * 0.4 + m_full['_calmar'] * 0.4 + m_full['_cagr'] * 0.2
    print(f"  Score = Sharpe×0.4 + Calmar×0.4 + CAGR/100×0.2")
    print(f"  = {m_full['_sharpe']:.2f}×0.4 + {m_full['_calmar']:.2f}×0.4 + {m_full['_cagr']:.3f}×0.2")
    print(f"  = {score:.3f}")
    
    # Weight analysis
    print()
    print("=" * 70)
    print("AVERAGE WEIGHTS")
    print("=" * 70)
    wdf = pd.DataFrame(wh_full)
    for col in ['QQQ', 'TLT', 'GLD', 'CASH']:
        print(f"  {col}: {wdf[col].mean():.1%} (min {wdf[col].min():.1%}, max {wdf[col].max():.1%})")
    
    return m_full, m_test, m_train


if __name__ == "__main__":
    run_walkforward()
