"""Fair comparison: CodeBear v5 vs 小袋熊 Beast 3Q80 — same data, same capital"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd, numpy as np

# Load data
df = pd.read_csv('data_cache/TQQQ.csv', parse_dates=['Date']).set_index('Date').sort_index()
prices = df['Close'].dropna()

# ── Indicators ──
sma200 = prices.rolling(200).mean()

# RSI helper
def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

rsi10 = rsi(prices, 10)
weekly_ret = prices.pct_change(5)

# ── v5 Beast (CodeBear) ──
def run_v5(prices, sma200, rsi10, weekly_ret, capital=10000):
    cash = capital; shares = 0
    regime = 'bull'
    values = []
    for i in range(len(prices)):
        p = prices.iloc[i]
        sma = sma200.iloc[i]
        r = rsi10.iloc[i]
        wr = weekly_ret.iloc[i]
        
        if pd.isna(sma):
            values.append(cash + shares * p)
            continue
        
        # Regime with hysteresis
        if regime == 'bull' and p < sma * 0.90:
            regime = 'bear'
        elif regime == 'bear' and p > sma * 1.05:
            regime = 'bull'
        
        if regime == 'bull':
            if not pd.isna(r) and r > 80 and not pd.isna(wr) and wr > 0.15:
                target = 0.80
            else:
                target = 1.0
        else:  # bear
            if not pd.isna(r) and r < 20:
                target = 0.80
            elif not pd.isna(r) and r < 30:
                target = 0.60
            elif not pd.isna(wr) and wr < -0.12:
                target = 0.80
            elif not pd.isna(r) and r > 65:
                target = 0.30
            else:
                target = 0.30
        
        curr = cash + shares * p
        target_eq = curr * target
        diff = target_eq - shares * p
        if diff > 0:
            buy = min(diff, cash)
            shares += buy / p; cash -= buy
        elif diff < 0:
            sell = min(abs(diff), shares * p)
            shares -= sell / p; cash += sell
        values.append(cash + shares * p)
    return pd.Series(values, index=prices.index)

# ── 小袋熊 Beast 3Q80 ──
def run_3q80(prices, sma200, rsi10, weekly_ret, capital=10000):
    cash = capital; shares = 0
    values = []
    for i in range(len(prices)):
        p = prices.iloc[i]
        sma = sma200.iloc[i]
        r = rsi10.iloc[i]
        wr = weekly_ret.iloc[i]
        
        if pd.isna(sma):
            values.append(cash + shares * p)
            continue
        
        if p > sma:  # Bull
            if not pd.isna(wr) and wr < -0.03:
                target = 1.0
            else:
                target = 0.80
        else:  # Bear
            if not pd.isna(wr) and wr < -0.10:
                target = 1.0
            elif not pd.isna(r) and r < 20:
                target = 1.0
            elif not pd.isna(r) and r < 30:
                target = 0.80
            else:
                target = 0.0
        
        curr = cash + shares * p
        target_eq = curr * target
        diff = target_eq - shares * p
        if diff > 0:
            buy = min(diff, cash)
            shares += buy / p; cash -= buy
        elif diff < 0:
            sell = min(abs(diff), shares * p)
            shares -= sell / p; cash += sell
        values.append(cash + shares * p)
    return pd.Series(values, index=prices.index)

# ── Buy & Hold ──
def run_bh(prices, capital=10000):
    shares = capital / prices.iloc[0]
    return prices * shares

# Run all
v5 = run_v5(prices, sma200, rsi10, weekly_ret)
q80 = run_3q80(prices, sma200, rsi10, weekly_ret)
bh = run_bh(prices)

def metrics(s, name):
    years = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1/years) - 1
    peak = s.cummax()
    dd = (s - peak) / peak
    mdd = dd.min()
    daily_ret = s.pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    print(f"{name:25s} | Final: ${s.iloc[-1]:>12,.0f} | CAGR: {cagr*100:6.1f}% | MaxDD: {mdd*100:6.1f}% | Sharpe: {sharpe:.2f} | Calmar: {calmar:.2f}")

print(f"\nTQQQ Fair Comparison — {prices.index[0].date()} to {prices.index[-1].date()}, $10,000 Lump Sum\n")
print(f"{'Strategy':25s} | {'Final':>14s} | {'CAGR':>8s} | {'MaxDD':>8s} | {'Sharpe':>7s} | {'Calmar':>7s}")
print("-" * 95)
metrics(v5, "🐻 CodeBear v5")
metrics(q80, "🐨 小袋熊 3Q80")
metrics(bh, "📈 Buy & Hold")
