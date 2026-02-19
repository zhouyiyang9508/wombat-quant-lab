"""
小袋熊量化实验室 - TQQQ Ultimate Wombat v3.0 (ML增强)
Optimized by: 代码熊 🐻
新增：Walk-forward GradientBoosting Regime分类器 + MACD金叉信号
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Data ──────────────────────────────────────────────
def fetch_data(ticker="TQQQ", start="2010-01-01"):
    df = yf.download(ticker, start=start, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[['Close']].dropna().copy()
    df.columns = ['price']
    return df

# ── Technical Indicators (no look-ahead) ─────────────
def add_features(df):
    p = df['price']
    # RSI 14
    delta = p.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # MACD
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    df['macd_diff'] = ema12 - ema26
    df['macd_signal'] = df['macd_diff'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_diff'] - df['macd_signal']

    # Bollinger position
    ma20 = p.rolling(20).mean()
    std20 = p.rolling(20).std()
    df['bb_pos'] = (p - (ma20 - 2*std20)) / ((ma20 + 2*std20) - (ma20 - 2*std20)).replace(0, np.nan)

    # ATR ratio
    df['atr14'] = p.diff().abs().rolling(14).mean()
    df['atr_ratio'] = df['atr14'] / p

    # Volume proxy (use price change magnitude as volume proxy since TQQQ volume can be spotty)
    df['vol_ratio'] = p.pct_change().abs().rolling(5).mean() / p.pct_change().abs().rolling(20).mean().replace(0, np.nan)

    # MA ratios
    df['ma20'] = p.rolling(20).mean()
    df['ma50'] = p.rolling(50).mean()
    df['ma200'] = p.rolling(200).mean()
    df['ma20_50'] = df['ma20'] / df['ma50'].replace(0, np.nan)
    df['ma50_200'] = df['ma50'] / df['ma200'].replace(0, np.nan)

    # ADX 14
    high_proxy = p.rolling(2).max()
    low_proxy = p.rolling(2).min()
    tr = (high_proxy - low_proxy).clip(lower=p.diff().abs())
    atr = tr.rolling(14).mean()
    up = high_proxy.diff()
    dn = -low_proxy.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['adx'] = dx.rolling(14).mean()

    # Stochastic 14
    low14 = p.rolling(14).min()
    high14 = p.rolling(14).max()
    df['stoch'] = 100 * (p - low14) / (high14 - low14).replace(0, np.nan)

    # MA cross
    df['ma_cross'] = (df['ma20'] > df['ma50']).astype(float)

    # Label: 20-day forward return > 0 (shifted properly)
    df['fwd_ret_20'] = p.pct_change(20).shift(-20)
    df['label'] = (df['fwd_ret_20'] > 0).astype(int)

    return df

FEATURE_COLS = ['rsi', 'macd_diff', 'bb_pos', 'atr_ratio', 'vol_ratio',
                'ma20_50', 'ma50_200', 'adx', 'stoch', 'ma_cross']

# ── Walk-forward ML ──────────────────────────────────
def walk_forward_predict(df, min_train=500, retrain_every=63, horizon=20):
    """Walk-forward: at time T, train on [0 : T-horizon] labels only."""
    n = len(df)
    preds = pd.Series(np.nan, index=df.index)
    probs = pd.Series(np.nan, index=df.index)

    feat = df[FEATURE_COLS].values
    labels = df['label'].values

    model = None
    last_train = -retrain_every  # force first train

    for t in range(min_train, n):
        # Retrain periodically
        if t - last_train >= retrain_every:
            # Only use labels up to t-horizon to avoid label leakage
            train_end = t - horizon
            if train_end < 100:
                continue
            X_tr = feat[:train_end]
            y_tr = labels[:train_end]
            # Drop NaN rows
            mask = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
            if mask.sum() < 50:
                continue
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            model.fit(X_tr[mask], y_tr[mask])
            last_train = t

        if model is not None and not np.isnan(feat[t]).any():
            prob = model.predict_proba(feat[t:t+1])[0]
            bull_prob = prob[1] if len(prob) > 1 else prob[0]
            preds.iloc[t] = 1 if bull_prob > 0.5 else 0
            probs.iloc[t] = bull_prob

    return preds, probs

# ── Backtest ─────────────────────────────────────────
def backtest(df, preds, probs):
    prices = df['price'].values
    ma200 = df['ma200'].values
    macd_hist = df['macd_hist'].values
    dates = df.index

    cash = 10000.0
    shares = 0.0
    total_invested = 10000.0
    weekly_dca = 1000

    portfolio = np.zeros(len(dates))
    last_friday_price = prices[0]
    last_month = dates[0].month

    for i in range(len(dates)):
        date = dates[i]
        price = prices[i]
        ma = ma200[i]
        bull_prob = probs.iloc[i] if not np.isnan(probs.iloc[i]) else 0.5
        mh = macd_hist[i] if not np.isnan(macd_hist[i]) else 0

        total_val = cash + shares * price

        is_friday = (date.weekday() == 4)
        is_bear_ma = (i >= 200 and not np.isnan(ma) and price < ma)

        # ── Position sizing ──
        if is_bear_ma:
            target_ratio = 0.0
        elif bull_prob > 0.65:
            target_ratio = 1.0
        elif bull_prob > 0.5:
            target_ratio = 0.80
        else:
            target_ratio = 0.20

        # MACD golden cross bonus: if MACD hist crosses positive, nudge up
        if mh > 0 and (i > 0 and macd_hist[i-1] < 0 if i > 0 and not np.isnan(macd_hist[i-1]) else False):
            target_ratio = min(1.0, target_ratio + 0.1)

        # ── Weekly DCA ──
        if is_friday:
            cash += weekly_dca
            total_invested += weekly_dca

            weekly_ret = (price - last_friday_price) / last_friday_price if last_friday_price > 0 else 0
            # 3Q dip buy
            if weekly_ret < -0.03 and cash > 0:
                shares += cash / price
                cash = 0
            last_friday_price = price

        # ── Quarterly rebalance ──
        is_quarter_start = (i > 0 and date.month != dates[i-1].month and date.month in [1, 4, 7, 10])
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

        # Bear mode: force liquidate
        if is_bear_ma and shares > 0:
            cash += shares * price
            shares = 0

        portfolio[i] = cash + shares * price

    return portfolio, total_invested

# ── Metrics ──────────────────────────────────────────
def compute_metrics(portfolio, dates, total_invested, label="TQQQ v3.0 ML"):
    pv = pd.Series(portfolio, index=dates)
    final = pv.iloc[-1]
    start = pv.iloc[0]
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = (final / start) ** (1/years) - 1 if years > 0 else 0

    peak = pv.cummax()
    dd = (pv - peak) / peak
    mdd = dd.min()

    dr = pv.pct_change().dropna()
    rf = 0.045 / 252
    excess = dr - rf
    sharpe = excess.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    downside = dr[dr < 0].std()
    sortino = excess.mean() / downside * np.sqrt(252) if downside > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    print(f"\n{'='*50}")
    print(f"🐻 {label} — Results")
    print(f"{'='*50}")
    print(f"Total Invested:    ${total_invested:>15,.2f}")
    print(f"Final Value:       ${final:>15,.2f}")
    print(f"CAGR:              {cagr*100:>14.2f}%")
    print(f"Max Drawdown:      {mdd*100:>14.2f}%")
    print(f"Sharpe Ratio:      {sharpe:>14.2f}")
    print(f"Sortino Ratio:     {sortino:>14.2f}")
    print(f"Calmar Ratio:      {calmar:>14.2f}")
    print(f"{'='*50}")

    return {'total_invested': total_invested, 'final_value': final, 'cagr': cagr,
            'max_dd': mdd, 'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar}

# ── Main ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🐻 TQQQ Ultimate Wombat v3.0 (ML Enhanced)")
    print("Loading data...")
    df = fetch_data()
    print(f"Data: {len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}")

    print("Computing features...")
    df = add_features(df)

    print("Running walk-forward ML (this may take a minute)...")
    preds, probs = walk_forward_predict(df)
    valid = probs.notna().sum()
    print(f"ML predictions available for {valid} days")

    print("Running backtest...")
    portfolio, total_invested = backtest(df, preds, probs)
    metrics = compute_metrics(portfolio, df.index, total_invested, "TQQQ Ultimate Wombat v3.0 (ML)")
