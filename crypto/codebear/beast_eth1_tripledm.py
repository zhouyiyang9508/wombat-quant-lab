"""
Crypto Beast ETH1 — Triple Momentum Rotation (BTC/ETH/GLD)
代码熊 🐻

核心思路：扩展 v7f 的 DualMom，加入 ETH 作为第三个选择
- 每月比较 BTC、ETH、GLD 的相对动量
- 选最强的那个，配置更高权重
- ETH 在 DeFi/NFT 周期时能提供额外 alpha

策略：
1. 计算 BTC、ETH、GLD 的 6M 动量
2. 选 Top 2 动量的资产，按动量强度加权
3. 如果最强的是加密货币（BTC/ETH），保持 70-85% 仓位
4. 如果最强的是 GLD，则防御性配置
5. 保留减半周期和 Mayer Multiple 的优化

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

class BeastETH1:
    """Triple Momentum Rotation — BTC vs ETH vs GLD."""
    
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
        
        # Calculate 6M and 3M momentum for all three
        mom6 = prices.pct_change(180)
        mom3 = prices.pct_change(90)
        
        # Blended momentum (50% 6M + 50% 3M)
        mom = 0.5 * mom6 + 0.5 * mom3
        
        # BTC SMA200 for Mayer Multiple
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
            
            m_btc = mom['BTC'].iloc[i]
            m_eth = mom['ETH'].iloc[i]
            m_gld = mom['GLD'].iloc[i]
            
            mm = mayer.iloc[i]
            hm, h_gain = halving_info(prices.index[i], p_btc, prices['BTC'])
            
            # Handle NaN momentum
            if pd.isna(m_btc):
                m_btc = 0.03
            if pd.isna(m_eth):
                m_eth = 0.03
            if pd.isna(m_gld):
                m_gld = 0.02
            
            # Rank by momentum
            assets = [('BTC', m_btc), ('ETH', m_eth), ('GLD', m_gld)]
            assets.sort(key=lambda x: x[1], reverse=True)
            
            top1, mom1 = assets[0]
            top2, mom2 = assets[1]
            
            # Allocation logic
            if mom1 > 0 and mom2 > 0:
                # Top 2 both positive
                if top1 in ['BTC', 'ETH']:
                    # Crypto winning
                    if top2 in ['BTC', 'ETH']:
                        # Both cryptos winning - weight by momentum
                        total_mom = mom1 + mom2
                        w1 = 0.60 * (mom1 / total_mom) + 0.30
                        w2 = 0.60 * (mom2 / total_mom) + 0.10
                        w_gld = 0.05
                    else:
                        # Top crypto + GLD
                        w1 = 0.75
                        w2 = 0.20
                        w_gld = w2 if top2 == 'GLD' else 0
                else:
                    # GLD winning (defensive)
                    w1 = 0.55  # GLD
                    w2 = 0.35  # Runner up
                    w_gld = w1
            elif mom1 > 0:
                # Only top 1 positive
                if top1 in ['BTC', 'ETH']:
                    w1 = 0.80
                    w2 = 0.10
                    w_gld = 0.10 if top1 == 'GLD' else 0.10
                else:
                    # GLD only positive
                    w1 = 0.60
                    w2 = 0.25
                    w_gld = w1
            else:
                # All negative - defensive
                w1 = 0.30
                w2 = 0.25
                w_gld = 0.40
            
            # Map weights
            target_btc = w1 if top1 == 'BTC' else (w2 if top2 == 'BTC' else 0.15)
            target_eth = w1 if top1 == 'ETH' else (w2 if top2 == 'ETH' else 0.10)
            target_gld = w_gld if w_gld > 0 else (w1 if top1 == 'GLD' else (w2 if top2 == 'GLD' else 0.05))
            
            # Halving cycle boost
            if hm is not None and hm <= 18:
                # Early cycle - boost crypto
                crypto_total = target_btc + target_eth
                if crypto_total < 0.60:
                    boost = 0.60 - crypto_total
                    target_btc += boost * 0.6
                    target_eth += boost * 0.4
                    target_gld = max(0.05, target_gld - boost)
            
            # Mayer bubble protection
            if not pd.isna(mm):
                if mm > 3.5:
                    # Extreme bubble
                    target_btc = min(target_btc, 0.25)
                    target_eth = min(target_eth, 0.20)
                    target_gld = max(target_gld, 0.35)
                elif mm > 3.0:
                    target_btc = min(target_btc, 0.40)
                    target_eth = min(target_eth, 0.30)
                    target_gld = max(target_gld, 0.20)
                elif mm > 2.4:
                    target_btc = min(target_btc, 0.55)
                    target_eth = min(target_eth, 0.35)
            
            # Late cycle gain-based
            if h_gain is not None:
                if h_gain > 5.0:
                    target_btc = min(target_btc, 0.30)
                    target_eth = min(target_eth, 0.25)
                    target_gld = max(target_gld, 0.30)
                elif h_gain > 3.0:
                    target_btc = min(target_btc, 0.45)
                    target_eth = min(target_eth, 0.35)
            
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
    import sys
    
    strategy = BeastETH1()
    strategy.load_data()
    strategy.run_backtest()
    
    metrics = strategy.get_metrics()
    
    print("\n=== Crypto Beast ETH1: Triple Momentum (BTC/ETH/GLD) ===")
    print(f"CAGR:       {metrics['cagr']*100:>6.1f}%")
    print(f"Max DD:     {metrics['max_dd']*100:>6.1f}%")
    print(f"Sharpe:     {metrics['sharpe']:>6.2f}")
    print(f"Calmar:     {metrics['calmar']:>6.2f}")
    print(f"Composite:  {metrics['composite']:>6.3f}")
    
    # Save returns for portfolio testing
    returns = strategy._dr
    output_path = BASE / "crypto/codebear/beast_eth1_daily_returns.csv"
    returns_df = pd.DataFrame({'Date': returns.index, 'Return': returns.values})
    returns_df.to_csv(output_path, index=False)
    print(f"\nReturns saved to: {output_path}")

if __name__ == "__main__":
    main()
