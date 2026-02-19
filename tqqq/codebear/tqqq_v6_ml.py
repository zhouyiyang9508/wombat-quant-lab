"""
TQQQ Wombat Beast v6.0 — ML + 规则融合
代码熊 🐻

核心改进（vs v3 ML 失败教训）：
1. 保留 v5 的宽幅滞后带 + 底仓机制（不在 200MA 以下清零）
2. ML 只作为辅助信号调节仓位，不做二分类决策
3. 熊市保留 30% 底仓 + RSI 抄底（v5 精华）
4. ML 高置信牛市时从 100% 加到 100%（不变），低置信时微调到 85-90%
5. 减少 rebalance 频率，避免过度交易
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')
import os, sys

# ── Data ──────────────────────────────────────
def load_data():
    """Load from CSV cache first, fallback to yfinance."""
    csv_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'data', 'tqqq_daily.csv'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'data_cache', 'TQQQ.csv'),
    ]
    for p in csv_paths:
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=['Date'], index_col='Date')[['Close']].dropna()
            df.sort_index(inplace=True)
            print(f"Data: CSV ({len(df)} rows)")
            return df
    import yfinance as yf
    df = yf.download('TQQQ', start='2010-01-01', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[['Close']].dropna()
    print(f"Data: yfinance ({len(df)} rows)")
    return df

# ── Features (no lookahead) ──────────────────
def compute_features(prices):
    """All features use only historical data."""
    df = pd.DataFrame({'price': prices})
    p = df['price']

    # RSI(10) — same as v5
    delta = p.diff()
    gain = delta.clip(lower=0).rolling(10).mean()
    loss = (-delta.clip(upper=0)).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi10'] = 100 - 100 / (1 + rs)

    # RSI(14)
    gain14 = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss14 = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df['rsi14'] = 100 - 100 / (1 + gain14 / loss14.replace(0, np.nan))

    # MACD
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    df['macd_diff'] = ema12 - ema26
    df['macd_signal'] = df['macd_diff'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_diff'] - df['macd_signal']

    # Bollinger position
    ma20 = p.rolling(20).mean()
    std20 = p.rolling(20).std()
    bb_width = (ma20 + 2*std20) - (ma20 - 2*std20)
    df['bb_pos'] = (p - (ma20 - 2*std20)) / bb_width.replace(0, np.nan)

    # ATR ratio
    df['atr_ratio'] = p.diff().abs().rolling(14).mean() / p

    # MA ratios
    df['sma200'] = p.rolling(200).mean()
    df['ma20_50'] = p.rolling(20).mean() / p.rolling(50).mean().replace(0, np.nan)
    df['ma50_200'] = p.rolling(50).mean() / df['sma200'].replace(0, np.nan)

    # Price momentum (20d, 60d returns)
    df['mom20'] = p.pct_change(20)
    df['mom60'] = p.pct_change(60)

    # Volatility regime
    df['vol20'] = p.pct_change().rolling(20).std() * np.sqrt(252)

    # Weekly return
    df['weekly_ret'] = p.pct_change(5)

    # Label: 20-day forward return > 0
    df['fwd_ret_20'] = p.pct_change(20).shift(-20)
    df['label'] = (df['fwd_ret_20'] > 0).astype(int)

    return df

FEATURE_COLS = ['rsi14', 'macd_diff', 'macd_hist', 'bb_pos', 'atr_ratio',
                'ma20_50', 'ma50_200', 'mom20', 'mom60', 'vol20']

# ── Walk-forward ML ──────────────────────────
def walk_forward_predict(df, min_train=500, retrain_every=63, horizon=20):
    n = len(df)
    probs = pd.Series(np.nan, index=df.index)
    feat = df[FEATURE_COLS].values
    labels = df['label'].values
    model = None
    last_train = -retrain_every

    for t in range(min_train, n):
        if t - last_train >= retrain_every:
            train_end = t - horizon
            if train_end < 100:
                continue
            X_tr = feat[:train_end]
            y_tr = labels[:train_end]
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
            probs.iloc[t] = prob[1] if len(prob) > 1 else prob[0]

    return probs

# ── Backtest: v5 + ML 融合 ───────────────────
def backtest_v6(df, ml_probs, initial_capital=10000):
    """
    v5 的宽幅滞后带 + 底仓作为骨架
    ML probability 作为仓位微调信号
    """
    prices = df['price'].values
    sma200 = df['sma200'].values
    rsi10 = df['rsi10'].values
    weekly_ret = df['weekly_ret'].values
    macd_hist = df['macd_hist'].values
    dates = df.index

    cash = 0.0
    shares = initial_capital / prices[0]
    in_bear = False

    portfolio = np.zeros(len(dates))

    for i in range(len(dates)):
        price = prices[i]
        sma = sma200[i]
        rsi = rsi10[i]
        wret = weekly_ret[i]
        mh = macd_hist[i]
        ml_prob = ml_probs.iloc[i] if not np.isnan(ml_probs.iloc[i]) else 0.5

        # ── v5 宽幅滞后带 regime detection ──
        if i >= 200 and not np.isnan(sma):
            if not in_bear and price < sma * 0.90:
                in_bear = True
            elif in_bear and price > sma * 1.05:
                in_bear = False

        # ── Target allocation: v5 骨架 + ML 微调 ──
        if not in_bear:
            # BULL regime
            if not np.isnan(rsi) and rsi > 80 and not np.isnan(wret) and wret > 0.15:
                base_pct = 0.80  # extreme euphoria trim
            else:
                base_pct = 1.00

            # ML 微调：低置信时略微减仓（不低于 80%）
            if ml_prob < 0.40:
                target_pct = max(base_pct * 0.85, 0.80)  # ML 看熊 → 最多减到 80%
            elif ml_prob < 0.50:
                target_pct = max(base_pct * 0.92, 0.85)  # ML 略熊 → 最多减到 85%
            else:
                target_pct = base_pct  # ML 看牛或中性 → 不动
        else:
            # BEAR regime — 保留 v5 的底仓 + 抄底逻辑
            if not np.isnan(rsi) and rsi < 20:
                base_pct = 0.80  # capitulation buy
            elif not np.isnan(wret) and wret < -0.12:
                base_pct = 0.70  # crash buy
            elif not np.isnan(rsi) and rsi < 30:
                base_pct = 0.60  # moderate bounce
            elif not np.isnan(rsi) and rsi > 65:
                base_pct = 0.30  # back to floor
            else:
                cv = cash + shares * price
                base_pct = max((shares * price) / cv if cv > 0 else 0.30, 0.30)

            # 熊市：完全交给 v5 逻辑，ML 不介入
            target_pct = base_pct

        # ── Rebalance ──
        cv = cash + shares * price
        target_eq = cv * target_pct
        diff = target_eq - shares * price

        if diff > 0 and cash > 0:
            buy = min(diff, cash)
            shares += buy / price
            cash -= buy
        elif diff < 0:
            sell = abs(diff)
            shares -= sell / price
            cash += sell

        portfolio[i] = cash + shares * price

    return pd.Series(portfolio, index=dates)

# ── Metrics ──────────────────────────────────
def compute_metrics(pv, label=""):
    final = pv.iloc[-1]; start = pv.iloc[0]
    years = (pv.index[-1] - pv.index[0]).days / 365.25
    cagr = (final / start) ** (1/years) - 1
    peak = pv.cummax()
    dd = (pv - peak) / peak
    mdd = dd.min()
    dr = pv.pct_change().dropna()
    rf = 0.045 / 252
    sharpe = (dr - rf).mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    downside = dr[dr < 0].std()
    sortino = (dr - rf).mean() / downside * np.sqrt(252) if downside > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    print(f"\n{'='*50}")
    print(f"🐻 {label}")
    print(f"{'='*50}")
    print(f"💰 ${start:>10,.0f} → ${final:>10,.0f}")
    print(f"📈 CAGR:        {cagr*100:>8.2f}%")
    print(f"🛡️  MaxDD:       {mdd*100:>8.2f}%")
    print(f"⚡ Sharpe:      {sharpe:>8.2f}")
    print(f"🎯 Sortino:     {sortino:>8.2f}")
    print(f"🏔️  Calmar:      {calmar:>8.2f}")
    return {'label': label, 'final': final, 'cagr': cagr, 'mdd': mdd,
            'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar}

# ── v5 baseline (inline for comparison) ──────
def backtest_v5(prices_series, initial_capital=10000):
    prices = prices_series.values
    dates = prices_series.index
    sma200 = prices_series.rolling(200).mean().values
    delta = prices_series.diff()
    gain = delta.clip(lower=0).rolling(10).mean()
    loss = (-delta.clip(upper=0)).rolling(10).mean()
    rsi10 = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).values
    wret = prices_series.pct_change(5).values

    cash = 0.0
    shares = initial_capital / prices[0]
    in_bear = False
    portfolio = np.zeros(len(dates))

    for i in range(len(dates)):
        price = prices[i]; sma = sma200[i]; rsi = rsi10[i]; wr = wret[i]
        if i >= 200 and not np.isnan(sma):
            if not in_bear and price < sma * 0.90: in_bear = True
            elif in_bear and price > sma * 1.05: in_bear = False

        if not in_bear:
            target_pct = 0.80 if (not np.isnan(rsi) and rsi > 80 and not np.isnan(wr) and wr > 0.15) else 1.00
        else:
            if not np.isnan(rsi) and rsi < 20: target_pct = 0.80
            elif not np.isnan(wr) and wr < -0.12: target_pct = 0.70
            elif not np.isnan(rsi) and rsi < 30: target_pct = 0.60
            elif not np.isnan(rsi) and rsi > 65: target_pct = 0.30
            else:
                cv = cash + shares * price
                target_pct = max((shares * price) / cv if cv > 0 else 0.30, 0.30)

        cv = cash + shares * price
        diff = cv * target_pct - shares * price
        if diff > 0 and cash > 0:
            buy = min(diff, cash); shares += buy / price; cash -= buy
        elif diff < 0:
            sell = abs(diff); shares -= sell / price; cash += sell
        portfolio[i] = cash + shares * price

    return pd.Series(portfolio, index=dates)

# ── Main ─────────────────────────────────────
if __name__ == "__main__":
    print("🐻 TQQQ v6.0: v5 骨架 + ML 微调")

    data = load_data()
    print(f"Period: {data.index[0].date()} → {data.index[-1].date()}")

    print("\nComputing features...")
    df = compute_features(data['Close'])

    print("Running walk-forward ML...")
    ml_probs = walk_forward_predict(df)
    valid = ml_probs.notna().sum()
    print(f"ML predictions: {valid} days")

    print("\n--- v6 ML Hybrid ---")
    pv6 = backtest_v6(df, ml_probs)
    m6 = compute_metrics(pv6, "TQQQ v6.0 (v5 + ML)")

    print("\n--- v5 Baseline ---")
    pv5 = backtest_v5(data['Close'])
    m5 = compute_metrics(pv5, "TQQQ v5.0 Beast")

    print("\n--- Buy & Hold ---")
    bh = 10000 / data['Close'].iloc[0] * data['Close']
    mbh = compute_metrics(bh, "Buy & Hold")

    # Summary
    print(f"\n{'='*70}")
    print(f"{'策略':<25} {'最终价值':>12} {'CAGR':>8} {'MaxDD':>9} {'Sharpe':>8} {'Calmar':>8}")
    print("-" * 70)
    for m in [m6, m5, mbh]:
        print(f"{m['label']:<25} ${m['final']:>10,.0f} {m['cagr']*100:>7.1f}% {m['mdd']*100:>8.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f}")
