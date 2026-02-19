"""
公平对比：v5 vs v3 ML（都用 lump sum $10000）
代码熊 🐻
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np

# ── Helper: compute metrics ──
def metrics(portfolio_series, label=""):
    r = portfolio_series
    final = r.iloc[-1]; start = r.iloc[0]
    years = (r.index[-1] - r.index[0]).days / 365.25
    cagr = (final / start) ** (1/years) - 1 if years > 0 and start > 0 else 0
    peak = r.cummax()
    dd = (r - peak) / peak
    mdd = dd.min()
    dr = r.pct_change().dropna()
    rf = 0.045 / 252
    excess = dr - rf
    sharpe = excess.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    return {'label': label, 'final': final, 'cagr': cagr, 'mdd': mdd, 'sharpe': sharpe, 'calmar': calmar}

def print_comparison(results):
    print(f"\n{'策略':<35} {'最终价值':>12} {'CAGR':>8} {'MaxDD':>9} {'Sharpe':>8} {'Calmar':>8}")
    print("-" * 85)
    for r in results:
        print(f"{r['label']:<35} ${r['final']:>10,.0f} {r['cagr']*100:>7.1f}% {r['mdd']*100:>8.1f}% {r['sharpe']:>7.2f} {r['calmar']:>7.2f}")

# ══════════════════════════════════════════════
# TQQQ Comparison
# ══════════════════════════════════════════════
print("=" * 60)
print("🐻 TQQQ: v5 vs ML v3 (Lump Sum $10,000)")
print("=" * 60)

# Load data
csv_path = os.path.join(os.path.dirname(__file__), '..', 'tqqq', 'data', 'tqqq_daily.csv')
tqqq_data = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')[['Close']].dropna()
tqqq_data.sort_index(inplace=True)
print(f"Data source: CSV ({len(tqqq_data)} rows)")

print(f"Period: {tqqq_data.index[0].date()} → {tqqq_data.index[-1].date()}")

# --- TQQQ v5 ---
from tqqq.codebear.beast_v5 import WombatBeastV5
v5_tqqq = WombatBeastV5(initial_capital=10000)
v5_tqqq.data = tqqq_data.copy()
v5_tqqq.run_backtest()
r_v5_tqqq = metrics(v5_tqqq.results, "TQQQ v5 Beast")

# --- TQQQ Buy & Hold ---
bh_shares = 10000 / tqqq_data['Close'].iloc[0]
bh_series = bh_shares * tqqq_data['Close']
r_bh_tqqq = metrics(bh_series, "TQQQ Buy & Hold")

# --- TQQQ ML v3 (lump sum adaptation) ---
# We need to import and run the ML strategy in lump-sum mode
# Rewrite the backtest inline to use lump sum
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tqqq_v3_ml import add_features, walk_forward_predict, FEATURE_COLS
import warnings
warnings.filterwarnings('ignore')

# Use same data
df_ml = tqqq_data.copy()
df_ml.columns = ['price']
df_ml = add_features(df_ml)

print("Running walk-forward ML (may take a minute)...")
preds, probs = walk_forward_predict(df_ml)
valid = probs.notna().sum()
print(f"ML predictions: {valid} days")

# Lump sum backtest for ML v3
prices_ml = df_ml['price'].values
ma200_ml = df_ml['ma200'].values
macd_hist_ml = df_ml['macd_hist'].values
dates_ml = df_ml.index

cash = 0.0
shares = 10000.0 / prices_ml[0]  # start fully invested
portfolio_ml = np.zeros(len(dates_ml))
last_month = dates_ml[0].month

for i in range(len(dates_ml)):
    date = dates_ml[i]
    price = prices_ml[i]
    ma = ma200_ml[i]
    bull_prob = probs.iloc[i] if not np.isnan(probs.iloc[i]) else 0.5
    mh = macd_hist_ml[i] if not np.isnan(macd_hist_ml[i]) else 0

    is_bear_ma = (i >= 200 and not np.isnan(ma) and price < ma)

    if is_bear_ma:
        target_ratio = 0.0
    elif bull_prob > 0.65:
        target_ratio = 1.0
    elif bull_prob > 0.5:
        target_ratio = 0.80
    else:
        target_ratio = 0.20

    # MACD golden cross bonus
    if mh > 0 and (i > 0 and not np.isnan(macd_hist_ml[i-1]) and macd_hist_ml[i-1] < 0):
        target_ratio = min(1.0, target_ratio + 0.1)

    # Quarterly rebalance
    is_quarter_start = (i > 0 and date.month != dates_ml[i-1].month and date.month in [1, 4, 7, 10])
    if is_quarter_start:
        total_val = cash + shares * price
        target_eq = total_val * target_ratio
        current_eq = shares * price
        diff = target_eq - current_eq
        if diff > 0:
            buy = min(diff, cash)
            shares += buy / price
            cash -= buy
        elif diff < 0:
            sell = min(abs(diff), shares * price)
            shares -= sell / price
            cash += sell

    # Bear mode force liquidate
    if is_bear_ma and shares > 0:
        cash += shares * price
        shares = 0

    portfolio_ml[i] = cash + shares * price
    last_month = date.month

ml_series = pd.Series(portfolio_ml, index=dates_ml)
r_ml_tqqq = metrics(ml_series, "TQQQ v3 ML (lump sum)")

print_comparison([r_v5_tqqq, r_ml_tqqq, r_bh_tqqq])

# ══════════════════════════════════════════════
# BTC Comparison
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("🐻 BTC: v5 vs ML v3 (Lump Sum $10,000)")
print("=" * 60)

csv_path_btc = os.path.join(os.path.dirname(__file__), '..', 'btc', 'data', 'btc_daily.csv')
btc_data = pd.read_csv(csv_path_btc, parse_dates=['Date'], index_col='Date')[['Close']].dropna()
btc_data.sort_index(inplace=True)
print(f"Data source: CSV ({len(btc_data)} rows)")

# Filter to common start (2017+)
btc_data = btc_data[btc_data.index >= '2017-01-01'].copy()
print(f"Period: {btc_data.index[0].date()} → {btc_data.index[-1].date()}")

# --- BTC v5 ---
from btc.codebear.beast_v5 import BTCBeastV5
v5_btc = BTCBeastV5(initial_capital=10000)
v5_btc.data = btc_data.copy()
v5_btc.run_backtest()
r_v5_btc = metrics(v5_btc.results, "BTC v5 Beast")

# --- BTC Buy & Hold ---
bh_btc_shares = 10000 / btc_data['Close'].iloc[0]
bh_btc = bh_btc_shares * btc_data['Close']
r_bh_btc = metrics(bh_btc, "BTC Buy & Hold")

# --- BTC ML v3 (lump sum) ---
from btc_beast_v3_ml import add_indicators, multi_factor_score, score_to_ratio, halving_multiplier, is_quarter_end_week

df_btc_ml = btc_data.copy()
df_btc_ml.columns = ['price']

# Need full history for indicators
btc_full = pd.read_csv(csv_path_btc, parse_dates=['Date'], index_col='Date')[['Close']].dropna()
btc_full.sort_index(inplace=True)
btc_full.columns = ['price']

df_btc_ml = add_indicators(btc_full, start_date='2017-01-01')

# Lump sum backtest
ATH_DD_GUARD = 0.60
CIRCUIT_BREAKER = 0.30
cash_b = 0.0
btc_b = 10000.0 / df_btc_ml['price'].iloc[0]
last_q_val = 10000.0
values_b = []

for date, row in df_btc_ml.iterrows():
    price = row['price']
    ath_dd = row['ath_dd']
    hm = row['halving_months']

    val = cash_b + btc_b * price

    mf_score = multi_factor_score(row)
    mf_ratio = score_to_ratio(mf_score)

    pi = row.get('pi_ratio', 0.8)
    if not np.isnan(pi) and pi > 0.95:
        mf_ratio = min(mf_ratio, 0.10)

    # ATH drawdown protection
    if ath_dd < -ATH_DD_GUARD and btc_b > 0:
        target_btc_val = val * 0.20
        current_btc_val = btc_b * price
        if current_btc_val > target_btc_val:
            sell = current_btc_val - target_btc_val
            btc_b -= sell / price
            cash_b += sell

    # Quarterly rebalance
    if is_quarter_end_week(date):
        val = cash_b + btc_b * price
        is_crash = (last_q_val > 0 and (val - last_q_val) / last_q_val < -CIRCUIT_BREAKER)
        target_btc_val = val * mf_ratio
        current_btc_val = btc_b * price
        if current_btc_val > target_btc_val:
            sell = current_btc_val - target_btc_val
            btc_b -= sell / price
            cash_b += sell
        elif not is_crash:
            buy = min(target_btc_val - current_btc_val, cash_b)
            btc_b += buy / price
            cash_b -= buy
        last_q_val = val

    values_b.append(cash_b + btc_b * price)

btc_ml_series = pd.Series(values_b, index=df_btc_ml.index)
r_ml_btc = metrics(btc_ml_series, "BTC v3 ML (lump sum)")

print_comparison([r_v5_btc, r_ml_btc, r_bh_btc])

# ══════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("🐻 FINAL VERDICT")
print("=" * 60)
all_results = {
    'tqqq': [r_v5_tqqq, r_ml_tqqq, r_bh_tqqq],
    'btc': [r_v5_btc, r_ml_btc, r_bh_btc],
}
for asset, results in all_results.items():
    best = max(results, key=lambda x: x['calmar'])
    print(f"\n{asset.upper()} 最优 (Calmar): {best['label']} — CAGR {best['cagr']*100:.1f}%, MaxDD {best['mdd']*100:.1f}%, Calmar {best['calmar']:.2f}")
