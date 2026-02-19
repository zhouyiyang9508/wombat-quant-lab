"""
Fair comparison: v5 vs v3 ML strategies (all Lump Sum $10000)
代码熊 🐻
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    HAS_YF = True
except:
    HAS_YF = False

from sklearn.ensemble import GradientBoostingClassifier

# ============================================================
# DATA LOADING
# ============================================================
def load_tqqq():
    csv = os.path.join(os.path.dirname(__file__), 'tqqq/data/tqqq_daily.csv')
    if os.path.exists(csv):
        df = pd.read_csv(csv, parse_dates=['Date'], index_col='Date')
        df = df[['Close']].dropna().sort_index()
        print(f"TQQQ from CSV: {len(df)} rows")
        return df
    if HAS_YF:
        df = yf.download('TQQQ', start='2010-01-01', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df[['Close']].dropna()
        print(f"TQQQ from yfinance: {len(df)} rows")
        return df
    raise RuntimeError("No TQQQ data")

def load_btc():
    csv = os.path.join(os.path.dirname(__file__), 'btc/data/btc_daily.csv')
    if os.path.exists(csv):
        df = pd.read_csv(csv, parse_dates=['Date'], index_col='Date')
        df = df[['Close']].dropna().sort_index()
        print(f"BTC from CSV: {len(df)} rows")
        return df
    if HAS_YF:
        df = yf.download('BTC-USD', start='2014-09-17', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df[['Close']].dropna()
        print(f"BTC from yfinance: {len(df)} rows")
        return df
    raise RuntimeError("No BTC data")

# ============================================================
# METRICS
# ============================================================
def calc_metrics(portfolio_series):
    r = portfolio_series
    final = r.iloc[-1]; start = r.iloc[0]
    years = (r.index[-1] - r.index[0]).days / 365.25
    cagr = (final / start) ** (1/years) - 1 if years > 0 else 0
    peak = r.cummax()
    dd = (r - peak) / peak
    max_dd = dd.min()
    dr = r.pct_change().dropna()
    rf = 0.045 / 252
    sharpe = ((dr - rf).mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe, 'calmar': calmar,
            'final': final, 'start': start}

# ============================================================
# TQQQ v5
# ============================================================
def run_tqqq_v5(data, capital=10000):
    prices = data['Close']
    sma200 = prices.rolling(200).mean()
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(10).mean()
    loss = (-delta.clip(upper=0)).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi10 = 100 - (100 / (1 + rs))
    weekly_ret = prices.pct_change(5)

    cash = capital; shares = 0; in_bear = False; pvs = []
    for i in range(len(prices)):
        price = prices.iloc[i]; sma = sma200.iloc[i]
        rsi = rsi10.iloc[i]; wret = weekly_ret.iloc[i]
        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * 0.90: in_bear = True
            elif in_bear and price > sma * 1.05: in_bear = False
        if not in_bear:
            if not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15:
                target_pct = 0.80
            else: target_pct = 1.00
        else:
            if not pd.isna(rsi) and rsi < 20: target_pct = 0.80
            elif not pd.isna(wret) and wret < -0.12: target_pct = 0.70
            elif not pd.isna(rsi) and rsi < 30: target_pct = 0.60
            elif not pd.isna(rsi) and rsi > 65: target_pct = 0.30
            else:
                cv = cash + shares * price
                target_pct = max((shares * price) / cv if cv > 0 else 0.30, 0.30)
        cv = cash + shares * price
        diff = cv * target_pct - shares * price
        if diff > 0 and cash > 0:
            buy = min(diff, cash); shares += buy/price; cash -= buy
        elif diff < 0:
            sell = abs(diff); shares -= sell/price; cash += sell
        pvs.append(cash + shares * price)
    return pd.Series(pvs, index=prices.index)

# ============================================================
# TQQQ v3 ML (Lump Sum)
# ============================================================
def run_tqqq_v3_ml(data, capital=10000):
    df = data.copy()
    df.columns = ['price']
    p = df['price']

    # Features
    delta = p.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    df['macd_diff'] = ema12 - ema26
    df['macd_signal'] = df['macd_diff'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_diff'] - df['macd_signal']
    ma20 = p.rolling(20).mean(); std20 = p.rolling(20).std()
    df['bb_pos'] = (p - (ma20 - 2*std20)) / ((ma20 + 2*std20) - (ma20 - 2*std20)).replace(0, np.nan)
    df['atr_ratio'] = p.diff().abs().rolling(14).mean() / p
    df['vol_ratio'] = p.pct_change().abs().rolling(5).mean() / p.pct_change().abs().rolling(20).mean().replace(0, np.nan)
    df['ma20'] = ma20; df['ma50'] = p.rolling(50).mean(); df['ma200'] = p.rolling(200).mean()
    df['ma20_50'] = df['ma20'] / df['ma50'].replace(0, np.nan)
    df['ma50_200'] = df['ma50'] / df['ma200'].replace(0, np.nan)
    high_proxy = p.rolling(2).max(); low_proxy = p.rolling(2).min()
    tr = (high_proxy - low_proxy).clip(lower=p.diff().abs())
    atr = tr.rolling(14).mean()
    up = high_proxy.diff(); dn = -low_proxy.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['adx'] = dx.rolling(14).mean()
    low14 = p.rolling(14).min(); high14 = p.rolling(14).max()
    df['stoch'] = 100 * (p - low14) / (high14 - low14).replace(0, np.nan)
    df['ma_cross'] = (df['ma20'] > df['ma50']).astype(float)
    df['fwd_ret_20'] = p.pct_change(20).shift(-20)
    df['label'] = (df['fwd_ret_20'] > 0).astype(int)

    FEAT = ['rsi', 'macd_diff', 'bb_pos', 'atr_ratio', 'vol_ratio',
            'ma20_50', 'ma50_200', 'adx', 'stoch', 'ma_cross']

    # Walk-forward ML
    feat = df[FEAT].values; labels = df['label'].values
    n = len(df); probs = np.full(n, np.nan)
    model = None; last_train = -63
    for t in range(500, n):
        if t - last_train >= 63:
            train_end = t - 20
            if train_end < 100: continue
            X_tr = feat[:train_end]; y_tr = labels[:train_end]
            mask = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
            if mask.sum() < 50: continue
            model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)
            model.fit(X_tr[mask], y_tr[mask])
            last_train = t
        if model is not None and not np.isnan(feat[t]).any():
            prob = model.predict_proba(feat[t:t+1])[0]
            probs[t] = prob[1] if len(prob) > 1 else prob[0]

    # Backtest: LUMP SUM (no DCA)
    prices = df['price'].values
    ma200 = df['ma200'].values
    macd_hist_vals = df['macd_hist'].values
    cash = capital; shares = 0.0; pvs = np.zeros(n)

    for i in range(n):
        price = prices[i]; ma = ma200[i]
        bull_prob = probs[i] if not np.isnan(probs[i]) else 0.5
        mh = macd_hist_vals[i] if not np.isnan(macd_hist_vals[i]) else 0
        is_bear_ma = (i >= 200 and not np.isnan(ma) and price < ma)

        if is_bear_ma:
            target_ratio = 0.0
        elif bull_prob > 0.65:
            target_ratio = 1.0
        elif bull_prob > 0.5:
            target_ratio = 0.80
        else:
            target_ratio = 0.20

        # MACD cross bonus
        if mh > 0 and i > 0 and not np.isnan(macd_hist_vals[i-1]) and macd_hist_vals[i-1] < 0:
            target_ratio = min(1.0, target_ratio + 0.1)

        # Rebalance daily (since no DCA, we rebalance based on signal changes)
        total_val = cash + shares * price
        target_eq = total_val * target_ratio
        current_eq = shares * price
        diff = target_eq - current_eq
        if diff > 0:
            buy = min(diff, cash); shares += buy/price; cash -= buy
        elif diff < 0:
            sell = min(abs(diff), shares * price); shares -= sell/price; cash += sell

        pvs[i] = cash + shares * price

    return pd.Series(pvs, index=df.index)

# ============================================================
# BTC v5
# ============================================================
HALVING_DATES = [pd.Timestamp('2012-11-28'), pd.Timestamp('2016-07-09'),
                 pd.Timestamp('2020-05-11'), pd.Timestamp('2024-04-20')]

def months_since_halving(date):
    past = [h for h in HALVING_DATES if h <= date]
    return (date - past[-1]).days / 30.44 if past else None

def halving_mult(hm):
    if hm is None: return 1.0
    if hm <= 6: return 1.0 + 0.5*(hm/6)
    elif hm <= 12: return 1.5
    elif hm <= 18: return 1.5 - 0.5*((hm-12)/6)
    elif hm <= 30: return 1.0
    elif hm <= 42: return 1.0 - 0.15*((hm-30)/12)
    else: return 0.85

def run_btc_v5(data, capital=10000):
    prices = data['Close']
    sma200 = prices.rolling(200).mean()
    mayer = prices / sma200
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi14 = 100 - (100 / (1 + rs))
    weekly_ret = prices.pct_change(7)
    hm_series = pd.Series([months_since_halving(d) for d in prices.index], index=prices.index)

    cash = capital; btc = 0; in_bear = False; pvs = []
    for i in range(len(prices)):
        price = prices.iloc[i]; sma = sma200.iloc[i]; mm = mayer.iloc[i]
        rsi = rsi14.iloc[i]; wret = weekly_ret.iloc[i]; hm = hm_series.iloc[i]
        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * 0.92: in_bear = True
            elif in_bear and price > sma * 1.03: in_bear = False
        h_mult = halving_mult(hm)
        if not in_bear:
            if not pd.isna(mm) and mm > 3.5: target_pct = 0.50
            elif not pd.isna(mm) and mm > 3.0: target_pct = 0.70
            elif not pd.isna(mm) and mm > 2.4: target_pct = 0.85
            else: target_pct = 1.00
            if hm is not None and hm > 30: target_pct = min(target_pct, 0.90)
        else:
            floor = 0.35; adjusted_floor = min(floor * h_mult, 0.50)
            if not pd.isna(rsi) and rsi < 20: target_pct = 0.80
            elif not pd.isna(wret) and wret < -0.20: target_pct = 0.70
            elif not pd.isna(rsi) and rsi < 30: target_pct = 0.60
            elif not pd.isna(rsi) and rsi > 60: target_pct = adjusted_floor
            else:
                cv = cash + btc * price
                target_pct = max((btc * price) / cv if cv > 0 else adjusted_floor, adjusted_floor)
        cv = cash + btc * price
        diff = cv * target_pct - btc * price
        if diff > 0 and cash > 0:
            buy = min(diff, cash); btc += buy/price; cash -= buy
        elif diff < 0:
            sell = abs(diff); btc -= sell/price; cash += sell
        pvs.append(cash + btc * price)
    return pd.Series(pvs, index=prices.index)

# ============================================================
# BTC v3 ML (Lump Sum)
# ============================================================
def run_btc_v3_ml(data, capital=10000):
    df = data.copy()
    df.columns = ['price']
    p = df['price']

    # Indicators
    log_p = np.log(p)
    df['geom_mean_200'] = np.exp(log_p.rolling(200).mean())
    genesis = pd.Timestamp('2009-01-03')
    df['days_since_genesis'] = (df.index - genesis).days
    df['exp_growth'] = 10 ** (2.68 + 0.00057 * df['days_since_genesis'])
    df['ahr999'] = (p / df['geom_mean_200']) * (p / df['exp_growth'])
    df['ath'] = p.cummax()
    df['ath_dd'] = (p - df['ath']) / df['ath']
    df['weekly_ret'] = p.pct_change(5)
    df['halving_months'] = df.index.map(months_since_halving)
    df['ma50'] = p.rolling(50).mean()
    df['ma350x2'] = p.rolling(350).mean() * 2
    df['pi_ratio'] = df['ma50'] / df['ma350x2'].replace(0, np.nan)
    df['ma365'] = p.rolling(365).mean()
    df['puell'] = p / df['ma365'].replace(0, np.nan)
    delta = p.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    df['macd_diff'] = ema12 - ema26
    df['macd_signal'] = df['macd_diff'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_diff'] - df['macd_signal']
    df['ma200'] = p.rolling(200).mean()

    # Filter to start from 2017 (need 365-day indicators)
    df = df[df.index >= '2017-01-01'].copy()

    def multi_factor_score(row):
        score = 0.0
        rsi = row.get('rsi', 50)
        if not np.isnan(rsi):
            if rsi < 30: score += 0.5
            elif rsi > 70: score -= 0.5
        mh = row.get('macd_hist', 0)
        if not np.isnan(mh):
            if mh > 0: score += 0.3
            else: score -= 0.3
        pi = row.get('pi_ratio', 0.8)
        if not np.isnan(pi):
            if pi > 0.95: score -= 1.0
            elif pi < 0.70: score += 0.5
        puell = row.get('puell', 1.0)
        if not np.isnan(puell):
            if puell > 4.0: score -= 0.8
            elif puell < 0.5: score += 0.5
        ahr = row.get('ahr999', 1.0)
        if not np.isnan(ahr):
            if ahr < 0.45: score += 0.5
            elif ahr > 5.0: score -= 0.5
        return np.clip(score, -2, 2)

    def score_to_ratio(s):
        return 0.1 + (s + 2) / 4 * 0.9

    # Backtest: LUMP SUM
    cash = capital; btc = 0.0; pvs = []
    for date, row in df.iterrows():
        price = row['price']
        ath_dd = row['ath_dd']
        mf_score = multi_factor_score(row)
        mf_ratio = score_to_ratio(mf_score)
        pi = row.get('pi_ratio', 0.8)
        if not np.isnan(pi) and pi > 0.95:
            mf_ratio = min(mf_ratio, 0.10)
        # ATH drawdown protection
        val = cash + btc * price
        if ath_dd < -0.60 and btc > 0:
            target_btc_val = val * 0.20
            if btc * price > target_btc_val:
                sell = btc * price - target_btc_val
                btc -= sell / price; cash += sell
        # Rebalance to target
        val = cash + btc * price
        target_btc_val = val * mf_ratio
        diff = target_btc_val - btc * price
        if diff > 0:
            buy = min(diff, cash); btc += buy/price; cash -= buy
        elif diff < 0:
            sell = min(abs(diff), btc * price); btc -= sell/price; cash += sell
        pvs.append(cash + btc * price)
    return pd.Series(pvs, index=df.index)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🐻 代码熊 - v5 vs ML v3 公平对比 (Lump Sum $10000)")
    print("=" * 60)

    # --- TQQQ ---
    print("\n📊 Loading TQQQ data...")
    tqqq_data = load_tqqq()
    print(f"   {tqqq_data.index[0].date()} → {tqqq_data.index[-1].date()}")

    print("Running TQQQ v5...")
    tqqq_v5 = run_tqqq_v5(tqqq_data)
    m_v5 = calc_metrics(tqqq_v5)

    print("Running TQQQ v3 ML (lump sum)...")
    tqqq_v3 = run_tqqq_v3_ml(tqqq_data)
    m_v3 = calc_metrics(tqqq_v3)

    # Align dates for fair comparison
    common_start = max(tqqq_v5.index[0], tqqq_v3.index[0])
    common_end = min(tqqq_v5.index[-1], tqqq_v3.index[-1])
    # Recalc on common period
    tqqq_v5_c = tqqq_v5[(tqqq_v5.index >= common_start) & (tqqq_v5.index <= common_end)]
    tqqq_v3_c = tqqq_v3[(tqqq_v3.index >= common_start) & (tqqq_v3.index <= common_end)]
    # Normalize to same start
    tqqq_v5_n = tqqq_v5_c / tqqq_v5_c.iloc[0] * 10000
    tqqq_v3_n = tqqq_v3_c / tqqq_v3_c.iloc[0] * 10000
    m_v5_c = calc_metrics(tqqq_v5_n)
    m_v3_c = calc_metrics(tqqq_v3_n)

    print(f"\n{'='*60}")
    print(f"TQQQ 对比 (共同区间 {common_start.date()} → {common_end.date()})")
    print(f"{'='*60}")
    print(f"{'指标':<15} {'v5 Beast':>12} {'v3 ML':>12} {'Winner':>10}")
    print(f"{'-'*49}")
    for k, name in [('cagr','CAGR'), ('max_dd','MaxDD'), ('sharpe','Sharpe'), ('calmar','Calmar')]:
        v5v = m_v5_c[k]; v3v = m_v3_c[k]
        if k == 'max_dd':
            w = 'v5' if v5v > v3v else 'v3 ML'
            print(f"{name:<15} {v5v*100:>11.2f}% {v3v*100:>11.2f}% {w:>10}")
        elif k in ('cagr','sharpe','calmar'):
            w = 'v5' if v5v > v3v else 'v3 ML'
            if k == 'cagr':
                print(f"{name:<15} {v5v*100:>11.2f}% {v3v*100:>11.2f}% {w:>10}")
            else:
                print(f"{name:<15} {v5v:>12.2f} {v3v:>12.2f} {w:>10}")

    # --- BTC ---
    print(f"\n\n📊 Loading BTC data...")
    btc_data = load_btc()
    print(f"   {btc_data.index[0].date()} → {btc_data.index[-1].date()}")

    print("Running BTC v5...")
    btc_v5 = run_btc_v5(btc_data)
    m_bv5 = calc_metrics(btc_v5)

    print("Running BTC v3 ML (lump sum)...")
    btc_v3 = run_btc_v3_ml(btc_data)
    m_bv3 = calc_metrics(btc_v3)

    # Align
    common_start_b = max(btc_v5.index[0], btc_v3.index[0])
    common_end_b = min(btc_v5.index[-1], btc_v3.index[-1])
    btc_v5_c = btc_v5[(btc_v5.index >= common_start_b) & (btc_v5.index <= common_end_b)]
    btc_v3_c = btc_v3[(btc_v3.index >= common_start_b) & (btc_v3.index <= common_end_b)]
    btc_v5_n = btc_v5_c / btc_v5_c.iloc[0] * 10000
    btc_v3_n = btc_v3_c / btc_v3_c.iloc[0] * 10000
    m_bv5_c = calc_metrics(btc_v5_n)
    m_bv3_c = calc_metrics(btc_v3_n)

    print(f"\n{'='*60}")
    print(f"BTC 对比 (共同区间 {common_start_b.date()} → {common_end_b.date()})")
    print(f"{'='*60}")
    print(f"{'指标':<15} {'v5 Beast':>12} {'v3 ML':>12} {'Winner':>10}")
    print(f"{'-'*49}")
    for k, name in [('cagr','CAGR'), ('max_dd','MaxDD'), ('sharpe','Sharpe'), ('calmar','Calmar')]:
        v5v = m_bv5_c[k]; v3v = m_bv3_c[k]
        if k == 'max_dd':
            w = 'v5' if v5v > v3v else 'v3 ML'
            print(f"{name:<15} {v5v*100:>11.2f}% {v3v*100:>11.2f}% {w:>10}")
        elif k == 'cagr':
            w = 'v5' if v5v > v3v else 'v3 ML'
            print(f"{name:<15} {v5v*100:>11.2f}% {v3v*100:>11.2f}% {w:>10}")
        else:
            w = 'v5' if v5v > v3v else 'v3 ML'
            print(f"{name:<15} {v5v:>12.2f} {v3v:>12.2f} {w:>10}")

    # Summary
    print(f"\n\n{'='*60}")
    print("📋 SUMMARY")
    print(f"{'='*60}")
    
    # TQQQ winner
    tqqq_score_v5 = m_v5_c['cagr'] * 0.5 + m_v5_c['calmar'] * 0.5 / 10
    tqqq_score_v3 = m_v3_c['cagr'] * 0.5 + m_v3_c['calmar'] * 0.5 / 10
    tqqq_winner = 'v5' if tqqq_score_v5 >= tqqq_score_v3 else 'v3 ML'
    print(f"TQQQ best: {tqqq_winner} (v5 score={tqqq_score_v5:.4f}, v3={tqqq_score_v3:.4f})")
    
    btc_score_v5 = m_bv5_c['cagr'] * 0.5 + m_bv5_c['calmar'] * 0.5 / 10
    btc_score_v3 = m_bv3_c['cagr'] * 0.5 + m_bv3_c['calmar'] * 0.5 / 10
    btc_winner = 'v5' if btc_score_v5 >= btc_score_v3 else 'v3 ML'
    print(f"BTC best:  {btc_winner} (v5 score={btc_score_v5:.4f}, v3={btc_score_v3:.4f})")
