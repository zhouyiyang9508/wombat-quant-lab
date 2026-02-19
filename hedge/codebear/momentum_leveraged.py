"""
Momentum Rotation v2 — Leveraged Variants + Drawdown Circuit Breaker
Author: 代码熊 🐻

Tests 1x / 1.5x / 2x leverage on top of v2 (RP Top3 + mid-term momentum).
Includes financing cost (5% annualized) and drawdown circuit breaker (-25% cut → -15% restore).
"""
import numpy as np
import pandas as pd
from momentum_utils import (download_data, monthly_returns, momentum_score,
                             backtest_metrics, TRADE_COST, TICKERS, RISK_FREE_RATE)

TOP_N = 3
FINANCING_RATE = 0.05  # annualized borrowing cost

def run_leveraged(prices, leverage=1.0, use_financing=True, circuit_breaker=True,
                  cb_cut=-0.25, cb_restore=-0.15):
    """Run v2 strategy with leverage, financing cost, and drawdown circuit breaker."""
    tradeable = [t for t in TICKERS if t != 'SHY' and t in prices.columns]
    all_tickers = tradeable + ['SHY']
    monthly_p = prices[all_tickers].resample('ME').last().dropna(how='all')
    
    scores = momentum_score(monthly_p[tradeable], weights=[0.2, 0.4, 0.3, 0.1])
    ret = monthly_returns(prices[all_tickers].dropna(how='all'))
    rolling_vol = ret[all_tickers].rolling(6).std() * np.sqrt(12)
    
    strat_returns = []
    prev_holdings = {}
    cum_val = 1.0
    peak = 1.0
    lev_active = leverage  # current effective leverage
    
    for i in range(13, len(ret)):
        date = ret.index[i]
        sig_date = ret.index[i-1]
        if sig_date not in scores.index:
            continue
        
        # Circuit breaker logic
        if circuit_breaker and leverage > 1.0:
            dd = cum_val / peak - 1
            if dd < cb_cut:
                lev_active = 1.0
            elif dd > cb_restore:
                lev_active = leverage
        
        row = scores.loc[sig_date].dropna()
        if len(row) == 0:
            strat_returns.append((date, 0.0))
            continue
        
        ranked = row.sort_values(ascending=False)
        selected = []
        for ticker in ranked.index[:TOP_N]:
            selected.append(ticker if ranked[ticker] > 0 else 'SHY')
        
        # Inverse vol weights
        weights = {}
        for t in selected:
            vol = rolling_vol.loc[sig_date, t] if sig_date in rolling_vol.index and t in rolling_vol.columns else 0.15
            if pd.isna(vol) or vol < 0.01:
                vol = 0.15
            weights[t] = weights.get(t, 0) + 1.0 / vol
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {t: w / total_w for t, w in weights.items()}
        
        # Scale by leverage
        weights = {t: w * lev_active for t, w in weights.items()}
        
        # Portfolio return
        month_ret = sum(weights.get(t, 0) * ret.loc[date, t]
                        for t in weights if t in ret.columns and not pd.isna(ret.loc[date, t]))
        
        # Financing cost on borrowed portion
        if use_financing and lev_active > 1.0:
            borrowed = lev_active - 1.0
            # ~21 trading days per month
            month_ret -= borrowed * FINANCING_RATE / 12
        
        # Transaction costs
        turnover = sum(abs(weights.get(t, 0) - prev_holdings.get(t, 0))
                       for t in set(list(weights.keys()) + list(prev_holdings.keys()))) / 2
        month_ret -= turnover * TRADE_COST
        prev_holdings = weights.copy()
        
        strat_returns.append((date, month_ret))
        cum_val *= (1 + month_ret)
        peak = max(peak, cum_val)
    
    return pd.Series([r[1] for r in strat_returns], index=[r[0] for r in strat_returns])


def main():
    print("=" * 60)
    print("  Leveraged Momentum v2 — Comparison")
    print("=" * 60)
    
    prices = download_data()
    print(f"Data: {len(prices.columns)} assets, {len(prices)} days\n")
    
    configs = [
        ("1.0x (baseline)", 1.0, False, False),
        ("1.5x (no financing)", 1.5, False, False),
        ("1.5x (financing 5%)", 1.5, True, False),
        ("1.5x (fin + CB)", 1.5, True, True),
        ("2.0x (no financing)", 2.0, False, False),
        ("2.0x (financing 5%)", 2.0, True, False),
        ("2.0x (fin + CB)", 2.0, True, True),
    ]
    
    results = {}
    for name, lev, fin, cb in configs:
        ret = run_leveraged(prices, leverage=lev, use_financing=fin, circuit_breaker=cb)
        m = backtest_metrics(ret)
        results[name] = m
        print(f"{name:30s}  CAGR={m['CAGR']:6.2f}%  MaxDD={m['MaxDD']:7.2f}%  "
              f"Sharpe={m['Sharpe']:.2f}  Calmar={m['Calmar']:.2f}")
    
    # Return the monthly returns for monte carlo use
    ret_1x = run_leveraged(prices, 1.0, False, False)
    return results, ret_1x


if __name__ == '__main__':
    results, _ = main()
