"""
🐻 CodeBear BTC Beast v6 — Halving Cycle DCA Strategy

CORE INSIGHT:
  Bitcoin follows a predictable ~4-year halving cycle:
  - 0-12 months post-halving: accumulation → bull onset
  - 12-18 months: euphoria → cycle top
  - 18-30 months: bear market → capitulation
  - 30-48 months: deep value accumulation
  
  By SELLING during the 12-18 month peak zone (especially when Mayer > 2.0)
  and BUYING aggressively during the bear phase (especially on RSI < 25 or Mayer < 0.7),
  we capture the cycle swing.

RESULTS (2014-09-17 → 2026-02-18, $1,000/month DCA):
  📈 Buy & Hold:  $6,138,147 (44.5x) | MaxDD -82.9% | Sharpe 1.34
  🐻 v6 Beast:   $12,903,914 (93.5x) | MaxDD -66.1% | Sharpe 1.50
  → Beats B&H by +110% with 17% less drawdown

STRATEGY:
  Phase 1 (0-6 months post-halving): 100% BTC — accumulate
  Phase 2 (6-12 months): 100% BTC — early bull, stay fully invested
  Phase 3 (12-18 months): Mayer-dependent selling
    - Mayer > 2.5: 30% BTC (extreme bubble)
    - Mayer > 2.0: 50% BTC (bubble)
    - Mayer > 1.5: 70% BTC (elevated)
    - Otherwise: 90% BTC (cautious)
  Phase 4 (18-30 months): Bear market
    - RSI(14) < 25: 100% BTC (oversold bounce)
    - Mayer < 0.7: 100% BTC (deep value)
    - Otherwise: 50% BTC (minimal exposure)
  Phase 5 (30+ months): 100% BTC — deep accumulation

NO LOOK-AHEAD BIAS:
  - Halving dates are known events (not predictions)
  - Mayer Multiple uses trailing SMA200
  - RSI uses trailing 14-day data
  - All signals known at time of trade

HALVING DATES:
  - 2012-11-28 (Block 210,000)
  - 2016-07-09 (Block 420,000)
  - 2020-05-11 (Block 630,000)
  - 2024-04-20 (Block 840,000)
  - ~2028 (Block 1,050,000) — can extrapolate
"""

import pandas as pd
import numpy as np
import os

HALVING_DATES = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
    pd.Timestamp('2028-04-15'),  # estimated
]

class BeastV6BtcDCA:
    """BTC Beast v6 — Halving Cycle DCA Strategy"""
    
    def __init__(self, monthly_amount=1000):
        self.monthly_amount = monthly_amount
        self.data = None
    
    def load_data(self, csv_path=None):
        if csv_path is None:
            for path in ['data_cache/BTC_USD.csv', '../data_cache/BTC_USD.csv', 'btc/data/btc_daily.csv']:
                if os.path.exists(path):
                    csv_path = path
                    break
        df = pd.read_csv(csv_path, parse_dates=['Date']).set_index('Date').sort_index()
        self.data = df['Close'].dropna()
        print(f"📊 Loaded {len(self.data)} days: {self.data.index[0].date()} → {self.data.index[-1].date()}")
    
    def _calc_indicators(self):
        p = self.data
        self.sma200 = p.rolling(200).mean()
        self.mayer = p / self.sma200
        
        delta = p.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        self.rsi14 = 100 - 100 / (1 + rs)
        
        self.ath = p.cummax()
        self.dd_from_ath = (p - self.ath) / self.ath
    
    @staticmethod
    def days_since_halving(date):
        for h in reversed(HALVING_DATES):
            if date >= h:
                return (date - h).days
        return 9999
    
    def get_target_allocation(self, date):
        """Core strategy: halving cycle phase + Mayer + RSI."""
        dsh = self.days_since_halving(date)
        mm = self.mayer.get(date, np.nan)
        r14 = self.rsi14.get(date, np.nan)
        dd = self.dd_from_ath.get(date, np.nan)
        
        # ── Phase 1: Early post-halving (0-6 months) ──
        if dsh < 180:
            return 1.0
        
        # ── Phase 2: Bull building (6-12 months) ──
        elif dsh < 365:
            return 1.0
        
        # ── Phase 3: Peak zone (12-18 months) ──
        elif dsh < 540:
            if not pd.isna(mm):
                if mm > 2.5: return 0.30  # extreme bubble
                elif mm > 2.0: return 0.50  # bubble
                elif mm > 1.5: return 0.70  # elevated
            return 0.90  # cautious
        
        # ── Phase 4: Bear market (18-30 months) ──
        elif dsh < 900:
            # Override: buy on extreme signals
            if not pd.isna(r14) and r14 < 25:
                return 1.0  # oversold bounce
            if not pd.isna(mm) and mm < 0.7:
                return 1.0  # deep value
            if not pd.isna(dd) and dd < -0.65:
                return 1.0  # massive crash
            return 0.50  # otherwise cautious
        
        # ── Phase 5: Deep accumulation (30+ months) ──
        else:
            return 1.0
    
    def run_backtest(self):
        self._calc_indicators()
        prices = self.data
        monthly_dates = set(
            prices.groupby(prices.index.to_period('M')).apply(lambda x: x.index[0])
        )
        
        shares = 0.0; cash = 0.0; total_invested = 0.0
        daily_values = []; trades = []
        
        for date in prices.index:
            p = prices.loc[date]
            if date in monthly_dates:
                cash += self.monthly_amount
                total_invested += self.monthly_amount
                pv = shares * p + cash
                
                target_pct = self.get_target_allocation(date)
                target_eq = pv * target_pct
                curr_eq = shares * p
                diff = target_eq - curr_eq
                
                if diff > 0 and cash >= diff:
                    shares += diff / p; cash -= diff
                elif diff > 0:
                    shares += cash / p; cash = 0
                elif diff < 0:
                    sell = abs(diff); shares -= sell / p; cash += sell
                
                trades.append({
                    'date': date, 'price': p,
                    'dsh': self.days_since_halving(date),
                    'target_pct': target_pct,
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
        
        avg_alloc = self.trades['target_pct'].mean()
        n_reduced = (self.trades['target_pct'] < 1.0).sum()
        
        print("\n" + "=" * 60)
        print("🐻 BTC Beast v6 — Halving Cycle DCA")
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
        print(f"  Months Reduced:   {n_reduced}/{len(self.trades)}")
        print(f"  Final BTC:        {self.final_shares:.4f}")
        print(f"  Final Cash:       ${self.final_cash:,.0f}")
        print("=" * 60)


if __name__ == "__main__":
    bot = BeastV6BtcDCA(monthly_amount=1000)
    bot.load_data()
    bot.run_backtest()
    bot.report()
    
    print("\n--- Buy & Hold DCA Comparison ---")
    bh = BeastV6BtcDCA(monthly_amount=1000)
    bh.load_data()
    bh.get_target_allocation = lambda date: 1.0
    bh.run_backtest()
    bh.report()
