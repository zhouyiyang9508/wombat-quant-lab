"""
小袋熊量化实验室 - BTC Beast v3.0 (ML多因子增强)
Optimized by: 代码熊 🐻
新增：Pi Cycle Top、Puell Multiple代理、多因子评分动态仓位
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

HALVING_DATES = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
]

def months_since_halving(date):
    past = [h for h in HALVING_DATES if h <= date]
    if not past:
        return None
    return (date - past[-1]).days / 30.44

def fetch_data():
    raw = yf.download('BTC-USD', start='2014-09-17', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    df = raw[['Close']].dropna().copy()
    df.columns = ['price']
    return df

def add_indicators(df, start_date='2017-01-01'):
    p = df['price']

    # Ahr999
    log_p = np.log(p)
    df['geom_mean_200'] = np.exp(log_p.rolling(200).mean())
    genesis = pd.Timestamp('2009-01-03')
    df['days_since_genesis'] = (df.index - genesis).days
    df['exp_growth'] = 10 ** (2.68 + 0.00057 * df['days_since_genesis'])
    df['ahr999'] = (p / df['geom_mean_200']) * (p / df['exp_growth'])

    # ATH tracking
    df['ath'] = p.cummax()
    df['ath_dd'] = (p - df['ath']) / df['ath']

    # Weekly return
    df['weekly_ret'] = p.pct_change(5)

    # Halving
    df['halving_months'] = df.index.map(months_since_halving)

    # ── NEW: Pi Cycle Top ──
    df['ma50'] = p.rolling(50).mean()
    df['ma350x2'] = p.rolling(350).mean() * 2
    df['pi_ratio'] = df['ma50'] / df['ma350x2'].replace(0, np.nan)

    # ── NEW: Puell Multiple proxy (price / 365-day MA) ──
    df['ma365'] = p.rolling(365).mean()
    df['puell'] = p / df['ma365'].replace(0, np.nan)

    # ── RSI(14) ──
    delta = p.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # ── MACD ──
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    df['macd_diff'] = ema12 - ema26
    df['macd_signal'] = df['macd_diff'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_diff'] - df['macd_signal']

    # Filter to backtest period
    df = df[df.index >= start_date].copy()
    return df

def multi_factor_score(row):
    """
    Composite score from -2 to +2 based on multiple factors.
    All inputs are current/historical values only.
    """
    score = 0.0

    # RSI
    rsi = row.get('rsi', 50)
    if not np.isnan(rsi):
        if rsi < 30: score += 0.5      # oversold = bullish
        elif rsi > 70: score -= 0.5    # overbought = bearish

    # MACD histogram
    mh = row.get('macd_hist', 0)
    if not np.isnan(mh):
        if mh > 0: score += 0.3
        else: score -= 0.3

    # Pi Cycle
    pi = row.get('pi_ratio', 0.8)
    if not np.isnan(pi):
        if pi > 0.95: score -= 1.0     # top signal
        elif pi < 0.70: score += 0.5   # bottom signal

    # Puell
    puell = row.get('puell', 1.0)
    if not np.isnan(puell):
        if puell > 4.0: score -= 0.8   # top warning
        elif puell < 0.5: score += 0.5  # bottom opportunity

    # Ahr999
    ahr = row.get('ahr999', 1.0)
    if not np.isnan(ahr):
        if ahr < 0.45: score += 0.5
        elif ahr > 5.0: score -= 0.5

    return np.clip(score, -2, 2)

def score_to_ratio(score):
    """Map score [-2, +2] to target BTC ratio [0.1, 1.0]."""
    # Linear: -2 → 0.1, +2 → 1.0
    return 0.1 + (score + 2) / 4 * 0.9

def is_quarter_end_week(date):
    qe = {3: 31, 6: 30, 9: 30, 12: 31}
    m, d = date.month, date.day
    if m not in qe:
        return False
    return d >= (qe[m] - 4)

def halving_multiplier(hm):
    if hm is None: return 1.0
    if hm < 12: return 1.3
    elif hm < 30: return 1.0
    else: return 0.7

def backtest(df):
    WEEKLY_DCA = 1000
    PUMP_LIMIT = 0.07
    MISSED_FORCE = 3
    CIRCUIT_BREAKER = 0.30
    ATH_DD_GUARD = 0.60

    cash = 0.0
    btc = 0.0
    total_invested = 0.0
    missed_weeks = 0
    last_q_val = 0.0

    values = []

    for date, row in df.iterrows():
        price = row['price']
        ahr = row['ahr999'] if not pd.isna(row.get('ahr999', np.nan)) else 1.0
        weekly_chg = row['weekly_ret'] if not pd.isna(row.get('weekly_ret', np.nan)) else 0.0
        ath_dd = row['ath_dd']
        hm = row['halving_months']

        val = cash + btc * price
        if last_q_val == 0: last_q_val = val

        # Multi-factor score
        mf_score = multi_factor_score(row)
        mf_ratio = score_to_ratio(mf_score)

        # Pi Cycle top override
        pi = row.get('pi_ratio', 0.8)
        if not np.isnan(pi) and pi > 0.95:
            mf_ratio = min(mf_ratio, 0.10)

        # ATH drawdown protection
        if ath_dd < -ATH_DD_GUARD and btc > 0:
            target_btc_val = val * 0.20
            current_btc_val = btc * price
            if current_btc_val > target_btc_val:
                sell = current_btc_val - target_btc_val
                btc -= sell / price
                cash += sell

        # Weekly DCA (Friday)
        if date.weekday() == 4:
            invest = WEEKLY_DCA
            should_buy = False

            if weekly_chg < PUMP_LIMIT:
                should_buy = True
                missed_weeks = 0
            else:
                missed_weeks += 1
                if missed_weeks >= MISSED_FORCE:
                    should_buy = True
                    missed_weeks = 0

            # Ahr999 bottom + Puell bottom → double down
            puell = row.get('puell', 1.0)
            if not np.isnan(puell) and puell < 0.5 and ahr < 0.45:
                should_buy = True
                invest *= 3  # triple down at deep bottom
            elif ahr < 0.45:
                should_buy = True
                invest *= 2

            invest *= halving_multiplier(hm)

            if should_buy:
                total_spend = invest + cash
                if total_spend > 0:
                    btc += total_spend / price
                    cash = 0
                total_invested += WEEKLY_DCA
            else:
                cash += invest
                total_invested += WEEKLY_DCA

        # Quarterly rebalance
        if is_quarter_end_week(date):
            val = cash + btc * price
            is_crash = (last_q_val > 0 and (val - last_q_val) / last_q_val < -CIRCUIT_BREAKER)

            # Use multi-factor ratio instead of pure Ahr999
            target_btc_val = val * mf_ratio
            current_btc_val = btc * price

            if current_btc_val > target_btc_val:
                sell = current_btc_val - target_btc_val
                btc -= sell / price
                cash += sell
            elif not is_crash:
                buy = min(target_btc_val - current_btc_val, cash)
                btc += buy / price
                cash -= buy

            last_q_val = val

        values.append(cash + btc * price)

    return np.array(values), total_invested

def compute_metrics(portfolio, dates, total_invested, label="BTC Beast v3.0"):
    pv = pd.Series(portfolio, index=dates)
    final = pv.iloc[-1]
    start = pv.iloc[0]
    years = (dates[-1] - dates[0]).days / 365.25
    roi = (final - total_invested) / total_invested if total_invested > 0 else 0
    cagr = (final / start) ** (1/years) - 1 if years > 0 and start > 0 else 0

    peak = pv.cummax()
    dd = (pv - peak) / peak
    mdd = dd.min()

    dr = pv.pct_change().dropna()
    rf = 0.045 / 252
    excess = dr - rf
    sharpe = excess.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    print(f"\n{'='*50}")
    print(f"🐻 {label} — Results")
    print(f"{'='*50}")
    print(f"Total Invested:    ${total_invested:>15,.2f}")
    print(f"Final Value:       ${final:>15,.2f}")
    print(f"Total Return:      {roi*100:>14.2f}%")
    print(f"CAGR:              {cagr*100:>14.2f}%")
    print(f"Max Drawdown:      {mdd*100:>14.2f}%")
    print(f"Sharpe Ratio:      {sharpe:>14.2f}")
    print(f"Calmar Ratio:      {calmar:>14.2f}")
    print(f"{'='*50}")

    return {'total_invested': total_invested, 'final_value': final, 'roi': roi,
            'cagr': cagr, 'max_dd': mdd, 'sharpe': sharpe, 'calmar': calmar}

if __name__ == "__main__":
    print("🐻 BTC Beast v3.0 (Multi-Factor Enhanced)")
    print("Loading data...")
    df = fetch_data()
    print(f"Data: {len(df)} rows")
    
    print("Computing indicators...")
    df = add_indicators(df)
    print(f"Backtest: {df.index[0].date()} → {df.index[-1].date()}")

    print("Running backtest...")
    portfolio, total_invested = backtest(df)
    metrics = compute_metrics(portfolio, df.index, total_invested, "BTC Beast v3.0 (Multi-Factor)")
