"""
Crypto Beast ETH2 — Crypto Internal DualMom (BTC vs ETH) with GLD Hedge
代码熊 🐻

核心思路：在加密货币内部进行 BTC/ETH 轮动，熊市切到 GLD
- Crypto 部分：BTC vs ETH 动量比较，选更强的
- 防御模式：当两者都弱时，切到 GLD
- 保留减半周期优化（BTC 为基准）

策略逻辑：
1. 比较 BTC 和 ETH 的 6M 动量
2. 牛市：100% 配置动量更强的那个（BTC 或 ETH）
3. 混合市场：70-30 混合持有
4. 熊市：切换到 50-70% GLD

Author: 代码熊 🐻 | 2026-02-20
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]

HALVING_DATES = [
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
]

def halving_info(date, price, prices_series):
    past = [h for h in HALVING_DATES if h <= date]
    if not past:
        return None, None
    last_h = past[-1]
    months = (date - last_h).days / 30.44
    mask = prices_series.index >= last_h
    if mask.any():
        h_price = prices_series.loc[mask].iloc[0]
        gain = (price / h_price) - 1.0
    else:
        gain = 0.0
    return months, gain

class BeastETH2:
    """Crypto Internal DualMom — BTC vs ETH with GLD hedge."""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.data = None
        self.results = None
    
    def load_data(self, start='2017-01-01', end='2026-02-20'):
        btc = pd.read_csv(BASE / "data_cache/BTC_USD.csv", parse_dates=['Date'], index_col='Date')
        btc = btc[['Close']].dropna().sort_index().loc[start:end]
        btc.columns = ['BTC']
        
        eth = pd.read_csv(BASE / "data_cache/ETH_USD.csv", parse_dates=['Date'], index_col='Date')
        eth = eth[['Close']].dropna().sort_index().loc[start:end]
        eth.columns = ['ETH']
        
        gld = pd.read_csv(BASE / "data_cache/GLD.csv", parse_dates=['Date'], index_col='Date')
        gld = gld[['Close']].dropna().sort_index()
        gld.columns = ['GLD']
        
        combined = btc.join(eth, how='left')
        combined = combined.join(gld, how='left')
        combined['ETH'] = combined['ETH'].ffill()
        combined['GLD'] = combined['GLD'].ffill()
        combined = combined.dropna()
        
        self.data = combined
        return self
    
    def run_backtest(self):
        prices = self.data
        
        # 6M and 3M momentum
        mom6_btc = prices['BTC'].pct_change(180)
        mom6_eth = prices['ETH'].pct_change(180)
        mom6_gld = prices['GLD'].pct_change(180)
        
        mom3_btc = prices['BTC'].pct_change(90)
        mom3_eth = prices['ETH'].pct_change(90)
        mom3_gld = prices['GLD'].pct_change(90)
        
        # Blended
        mom_btc = 0.5 * mom6_btc + 0.5 * mom3_btc
        mom_eth = 0.5 * mom6_eth + 0.5 * mom3_eth
        mom_gld = 0.5 * mom6_gld + 0.5 * mom3_gld
        
        # BTC SMA200 for Mayer
        btc_sma200 = prices['BTC'].rolling(200).mean()
        mayer = prices['BTC'] / btc_sma200
        
        cash = self.initial_capital
        btc = 0.0
        eth = 0.0
        gld = 0.0
        portfolio_values = []
        
        for i in range(len(prices)):
            p_btc = prices['BTC'].iloc[i]
            p_eth = prices['ETH'].iloc[i]
            p_gld = prices['GLD'].iloc[i]
            
            m_btc = mom_btc.iloc[i]
            m_eth = mom_eth.iloc[i]
            m_gld = mom_gld.iloc[i]
            
            mm = mayer.iloc[i]
            hm, h_gain = halving_info(prices.index[i], p_btc, prices['BTC'])
            
            # Handle NaN
            if pd.isna(m_btc):
                m_btc = 0.03
            if pd.isna(m_eth):
                m_eth = 0.03
            if pd.isna(m_gld):
                m_gld = 0.02
            
            # Crypto DualMom logic
            if m_btc > 0 and m_eth > 0:
                # Both cryptos positive
                if m_btc > m_eth:
                    # BTC stronger
                    ratio = m_btc / (m_btc + m_eth)
                    target_btc = 0.60 + ratio * 0.25
                    target_eth = 0.85 - target_btc
                    target_gld = 0.10
                else:
                    # ETH stronger
                    ratio = m_eth / (m_btc + m_eth)
                    target_eth = 0.60 + ratio * 0.25
                    target_btc = 0.85 - target_eth
                    target_gld = 0.10
            elif m_btc > 0 and m_eth <= 0:
                # Only BTC positive
                target_btc = 0.80
                target_eth = 0.05
                target_gld = 0.15
            elif m_eth > 0 and m_btc <= 0:
                # Only ETH positive
                target_eth = 0.80
                target_btc = 0.05
                target_gld = 0.15
            else:
                # Both negative - defensive
                if m_gld > 0:
                    target_btc = 0.20
                    target_eth = 0.15
                    target_gld = 0.60
                else:
                    target_btc = 0.25
                    target_eth = 0.20
                    target_gld = 0.40
            
            # Halving cycle boost (BTC-based)
            if hm is not None and hm <= 18:
                # Early cycle - boost crypto
                crypto_total = target_btc + target_eth
                if crypto_total < 0.65:
                    boost = 0.65 - crypto_total
                    # Favor BTC slightly in early halving cycle
                    target_btc += boost * 0.6
                    target_eth += boost * 0.4
                    target_gld = max(0.05, target_gld - boost)
            
            # Mayer protection (BTC-based)
            if not pd.isna(mm):
                if mm > 3.5:
                    target_btc = min(target_btc, 0.30)
                    target_eth = min(target_eth, 0.25)
                    target_gld = max(target_gld, 0.35)
                elif mm > 3.0:
                    target_btc = min(target_btc, 0.45)
                    target_eth = min(target_eth, 0.35)
                    target_gld = max(target_gld, 0.20)
                elif mm > 2.4:
                    target_btc = min(target_btc, 0.60)
                    target_eth = min(target_eth, 0.45)
            
            # Late cycle protection
            if h_gain is not None:
                if h_gain > 5.0:
                    target_btc = min(target_btc, 0.35)
                    target_eth = min(target_eth, 0.30)
                    target_gld = max(target_gld, 0.30)
                elif h_gain > 3.0:
                    target_btc = min(target_btc, 0.50)
                    target_eth = min(target_eth, 0.40)
            
            # Normalize
            total = target_btc + target_eth + target_gld
            if total > 1.0:
                scale = 1.0 / total
                target_btc *= scale
                target_eth *= scale
                target_gld *= scale
            
            # Rebalance BTC
            cv = cash + btc * p_btc + eth * p_eth + gld * p_gld
            target_btc_val = cv * target_btc
            diff_btc = target_btc_val - btc * p_btc
            if diff_btc > 0:
                buy = min(diff_btc, cash)
                btc += buy / p_btc
                cash -= buy
            elif diff_btc < 0:
                sell = abs(diff_btc)
                btc -= sell / p_btc
                cash += sell
            
            # Rebalance ETH
            cv = cash + btc * p_btc + eth * p_eth + gld * p_gld
            target_eth_val = cv * target_eth
            diff_eth = target_eth_val - eth * p_eth
            if diff_eth > 0:
                buy = min(diff_eth, cash)
                eth += buy / p_eth
                cash -= buy
            elif diff_eth < 0:
                sell = abs(diff_eth)
                eth -= sell / p_eth
                cash += sell
            
            # Rebalance GLD
            cv = cash + btc * p_btc + eth * p_eth + gld * p_gld
            target_gld_val = cv * target_gld
            diff_gld = target_gld_val - gld * p_gld
            if diff_gld > 0:
                buy = min(diff_gld, cash)
                gld += buy / p_gld
                cash -= buy
            elif diff_gld < 0:
                sell = abs(diff_gld)
                gld -= sell / p_gld
                cash += sell
            
            portfolio_values.append(cv)
        
        self.results = pd.Series(portfolio_values, index=prices.index)
        self._dr = self.results.pct_change().dropna()
        return self.results
    
    def get_metrics(self):
        r = self.results
        years = (r.index[-1] - r.index[0]).days / 365.25
        cagr = (r.iloc[-1] / r.iloc[0]) ** (1 / years) - 1
        dd = (r - r.cummax()) / r.cummax()
        max_dd = dd.min()
        dr = self._dr
        rf = 0.04 / 365
        sharpe = ((dr - rf).mean() / dr.std()) * np.sqrt(365) if dr.std() > 0 else 0
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        composite = sharpe * 0.4 + calmar * 0.4 + min(cagr, 1.0) * 0.2
        return {
            'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe,
            'calmar': calmar, 'composite': composite
        }

def main():
    strategy = BeastETH2()
    strategy.load_data()
    strategy.run_backtest()
    
    metrics = strategy.get_metrics()
    
    print("\n=== Crypto Beast ETH2: Internal DualMom (BTC vs ETH) ===")
    print(f"CAGR:       {metrics['cagr']*100:>6.1f}%")
    print(f"Max DD:     {metrics['max_dd']*100:>6.1f}%")
    print(f"Sharpe:     {metrics['sharpe']:>6.2f}")
    print(f"Calmar:     {metrics['calmar']:>6.2f}")
    print(f"Composite:  {metrics['composite']:>6.3f}")
    
    # Save returns
    returns = strategy._dr
    output_path = BASE / "crypto/codebear/beast_eth2_daily_returns.csv"
    returns_df = pd.DataFrame({'Date': returns.index, 'Return': returns.values})
    returns_df.to_csv(output_path, index=False)
    print(f"\nReturns saved to: {output_path}")

if __name__ == "__main__":
    main()
