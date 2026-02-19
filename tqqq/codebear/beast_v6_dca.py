"""
🐻 CodeBear TQQQ Beast v6 — Volatility-Targeted DCA Strategy

CORE INSIGHT:
  TQQQ vol is regime-dependent: low (~40-60%) in calm bulls, extreme (>100%) in crashes.
  By reducing exposure during high-vol periods and staying fully invested in calm periods,
  we avoid the worst of drawdowns while capturing most of the upside.

RESULTS (2010-02-11 → 2026-02-18, $1,000/month DCA):
  📈 Buy & Hold:  $7,265,603 (37.6x) | MaxDD -81.6% | Sharpe 1.14
  🐻 v6 Beast:   $11,691,645 (60.6x) | MaxDD -74.3% | Sharpe 1.21
  → Beats B&H by +60.9% with 7% less drawdown

STRATEGY:
  1. Compute 20-day annualized volatility (backward-looking, no look-ahead)
  2. Set target allocation based on vol thresholds:
     - Vol > 120%: 30% TQQQ (extreme panic, sell most)
     - Vol > 90%:  50% TQQQ (elevated risk)
     - Vol > 70%:  80% TQQQ (mild caution)
     - Vol ≤ 70%:  100% TQQQ (calm, full exposure)
  3. Override: if Mayer Multiple < 0.7, hold at least 80% (deep value)
  4. Override: if RSI(14) < 25, hold at least 85% (extreme oversold)
  5. Override: if drawdown from ATH > 60%, hold at least 90% (crash recovery)
  6. Rebalance monthly on first trading day

WHY IT WORKS:
  - High vol = bad news incoming → reduce exposure, preserve capital
  - Vol clusters: extreme vol today predicts more extreme vol tomorrow
  - Crash overrides: when it's ALREADY crashed hard, vol is lagging indicator,
    so we override to catch the V-shape rebound
  - Cash from selling high-vol periods gets redeployed at lower prices

NO LOOK-AHEAD BIAS:
  - 20-day vol uses only past returns
  - Mayer Multiple uses trailing SMA200
  - RSI uses trailing 14-day data
  - All signals known at time of trade

PRACTICAL NOTES:
  - Monthly rebalancing = ~12 trades/year, very practical
  - No leverage, no options, no margin
  - Cash earns 0% in backtest (conservative; real cash earns ~4-5%)
  - If cash earned interest, results would be even better
"""

import pandas as pd
import numpy as np
import os

class BeastV6DCA:
    """TQQQ Beast v6 — Volatility-Targeted DCA Strategy"""
    
    def __init__(self, monthly_amount=1000):
        self.monthly_amount = monthly_amount
        self.data = None
    
    def load_data(self, csv_path=None):
        """Load TQQQ price data from CSV."""
        if csv_path is None:
            # Try common paths
            for path in [
                'data_cache/TQQQ.csv',
                '../data_cache/TQQQ.csv',
                'tqqq/data/tqqq_daily.csv',
            ]:
                if os.path.exists(path):
                    csv_path = path
                    break
        
        df = pd.read_csv(csv_path, parse_dates=['Date']).set_index('Date').sort_index()
        self.data = df['Close'].dropna()
        print(f"📊 Loaded {len(self.data)} days: {self.data.index[0].date()} → {self.data.index[-1].date()}")
    
    def _calc_indicators(self):
        """Calculate all indicators (backward-looking only)."""
        p = self.data
        self.sma200 = p.rolling(200).mean()
        self.mayer = p / self.sma200
        
        # RSI(14)
        delta = p.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        self.rsi14 = 100 - 100 / (1 + rs)
        
        # 20-day annualized volatility
        self.vol20 = p.pct_change().rolling(20).std() * np.sqrt(252)
        
        # Drawdown from ATH
        ath = p.cummax()
        self.dd_from_ath = (p - ath) / ath
    
    def get_target_allocation(self, date):
        """
        Core strategy logic.
        Returns target % of portfolio in TQQQ (0.0 to 1.0).
        """
        v = self.vol20.get(date, np.nan)
        mm = self.mayer.get(date, np.nan)
        r = self.rsi14.get(date, np.nan)
        dd = self.dd_from_ath.get(date, np.nan)
        
        if pd.isna(v):
            return 1.0  # no vol data yet, stay fully invested
        
        # ── Base allocation: inverse vol thresholds ──
        if v > 1.20:
            alloc = 0.30  # extreme panic
        elif v > 0.90:
            alloc = 0.50  # elevated risk
        elif v > 0.70:
            alloc = 0.80  # mild caution
        else:
            alloc = 1.00  # calm market, full send
        
        # ── Override 1: Deep value (Mayer < 0.7) ──
        if not pd.isna(mm) and mm < 0.70:
            alloc = max(alloc, 0.80)
        
        # ── Override 2: Extreme oversold (RSI < 25) ──
        if not pd.isna(r) and r < 25:
            alloc = max(alloc, 0.85)
        
        # ── Override 3: Massive crash (DD > 60%) ──
        if not pd.isna(dd) and dd < -0.60:
            alloc = max(alloc, 0.90)
        
        return alloc
    
    def run_backtest(self):
        """Run DCA backtest with monthly contributions."""
        self._calc_indicators()
        
        prices = self.data
        monthly_dates = set(
            prices.groupby(prices.index.to_period('M')).apply(lambda x: x.index[0])
        )
        
        shares = 0.0
        cash = 0.0
        total_invested = 0.0
        daily_values = []
        trades = []
        
        for date in prices.index:
            p = prices.loc[date]
            
            if date in monthly_dates:
                cash += self.monthly_amount
                total_invested += self.monthly_amount
                portfolio_val = shares * p + cash
                
                target_pct = self.get_target_allocation(date)
                target_equity = portfolio_val * target_pct
                current_equity = shares * p
                diff = target_equity - current_equity
                
                if diff > 0 and cash >= diff:
                    shares += diff / p; cash -= diff
                elif diff > 0:
                    shares += cash / p; cash = 0
                elif diff < 0:
                    sell = abs(diff)
                    shares -= sell / p; cash += sell
                
                trades.append({
                    'date': date, 'price': p,
                    'target_pct': target_pct,
                    'vol': self.vol20.get(date, np.nan),
                    'mayer': self.mayer.get(date, np.nan),
                    'portfolio': shares * p + cash
                })
            
            daily_values.append((date, shares * p + cash, total_invested))
        
        self.history = pd.DataFrame(daily_values, columns=['Date', 'Value', 'Invested']).set_index('Date')
        self.trades = pd.DataFrame(trades)
        self.final_shares = shares
        self.final_cash = cash
        
        return self.history
    
    def report(self):
        """Print strategy report."""
        h = self.history
        final = h['Value'].iloc[-1]
        invested = h['Invested'].iloc[-1]
        mult = final / invested
        years = (h.index[-1] - h.index[0]).days / 365.25
        
        peak = h['Value'].cummax()
        dd = (h['Value'] - peak) / peak
        max_dd = dd.min()
        max_dd_date = dd.idxmin()
        
        monthly_vals = h['Value'].resample('ME').last().dropna()
        monthly_rets = monthly_vals.pct_change().dropna()
        sharpe = monthly_rets.mean() / monthly_rets.std() * np.sqrt(12)
        calmar_r = mult ** (1/years) - 1
        calmar = calmar_r / abs(max_dd)
        
        # Trade stats
        if len(self.trades) > 0:
            avg_alloc = self.trades['target_pct'].mean()
            min_alloc = self.trades['target_pct'].min()
            n_reduced = (self.trades['target_pct'] < 1.0).sum()
        
        print("\n" + "=" * 60)
        print("🐻 TQQQ Beast v6 — Volatility-Targeted DCA")
        print("=" * 60)
        print(f"  Period:        {h.index[0].date()} → {h.index[-1].date()} ({years:.1f} years)")
        print(f"  Monthly DCA:   ${self.monthly_amount:,}/month")
        print(f"  Total Invested: ${invested:,.0f}")
        print(f"  Final Value:   ${final:,.0f}")
        print(f"  Profit:        {mult:.1f}x ({(mult-1)*100:.0f}%)")
        print(f"  Max Drawdown:  {max_dd*100:.1f}% (at {max_dd_date.date()})")
        print(f"  Sharpe:        {sharpe:.2f}")
        print(f"  Calmar:        {calmar:.2f}")
        print(f"  ─────────────────────────────────────")
        print(f"  Avg Allocation:   {avg_alloc*100:.0f}%")
        print(f"  Min Allocation:   {min_alloc*100:.0f}%")
        print(f"  Months Reduced:   {n_reduced}/{len(self.trades)}")
        print(f"  Final Shares:     {self.final_shares:.2f}")
        print(f"  Final Cash:       ${self.final_cash:,.0f}")
        print("=" * 60)


if __name__ == "__main__":
    bot = BeastV6DCA(monthly_amount=1000)
    bot.load_data()
    bot.run_backtest()
    bot.report()
    
    # Compare with Buy & Hold DCA
    print("\n--- Buy & Hold DCA Comparison ---")
    bh = BeastV6DCA(monthly_amount=1000)
    bh.load_data()
    # Override to always buy
    bh.get_target_allocation = lambda date: 1.0
    bh.run_backtest()
    bh.report()
