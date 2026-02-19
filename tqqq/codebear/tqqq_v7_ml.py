"""
TQQQ Wombat Beast v7.0 — ML 辅助 Regime 切换
代码熊 🐻

思路：v5 的仓位逻辑完全不动（已证明最优）
ML 只做一件事：优化熊牛切换时机
- v5 用固定 90%/105% 滞后带
- v7 用 ML 概率来动态调整切换阈值
  - ML 强看熊（prob < 0.35）→ 更早进入熊市（95% 阈值）
  - ML 强看牛（prob > 0.65）→ 更早退出熊市（102% 阈值）
  - 其他情况 → 用 v5 默认值
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import warnings, os
warnings.filterwarnings('ignore')

def load_data():
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
    return df[['Close']].dropna()

def compute_features(prices):
    df = pd.DataFrame({'price': prices})
    p = df['price']
    delta = p.diff()

    # RSI(10)
    g10 = delta.clip(lower=0).rolling(10).mean()
    l10 = (-delta.clip(upper=0)).rolling(10).mean()
    df['rsi10'] = 100 - 100 / (1 + g10 / l10.replace(0, np.nan))

    # RSI(14)
    g14 = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    l14 = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df['rsi14'] = 100 - 100 / (1 + g14 / l14.replace(0, np.nan))

    # MACD
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    df['macd_diff'] = ema12 - ema26
    df['macd_hist'] = df['macd_diff'] - df['macd_diff'].ewm(span=9, adjust=False).mean()

    # Bollinger
    ma20 = p.rolling(20).mean(); std20 = p.rolling(20).std()
    bw = (ma20 + 2*std20) - (ma20 - 2*std20)
    df['bb_pos'] = (p - (ma20 - 2*std20)) / bw.replace(0, np.nan)

    df['atr_ratio'] = p.diff().abs().rolling(14).mean() / p
    df['sma200'] = p.rolling(200).mean()
    df['ma20_50'] = p.rolling(20).mean() / p.rolling(50).mean().replace(0, np.nan)
    df['ma50_200'] = p.rolling(50).mean() / df['sma200'].replace(0, np.nan)
    df['mom20'] = p.pct_change(20)
    df['mom60'] = p.pct_change(60)
    df['vol20'] = p.pct_change().rolling(20).std() * np.sqrt(252)
    df['weekly_ret'] = p.pct_change(5)

    df['fwd_ret_20'] = p.pct_change(20).shift(-20)
    df['label'] = (df['fwd_ret_20'] > 0).astype(int)
    return df

FEAT_COLS = ['rsi14', 'macd_diff', 'macd_hist', 'bb_pos', 'atr_ratio',
             'ma20_50', 'ma50_200', 'mom20', 'mom60', 'vol20']

def walk_forward(df, min_train=500, retrain_every=63, horizon=20):
    n = len(df)
    probs = pd.Series(np.nan, index=df.index)
    feat = df[FEAT_COLS].values
    labels = df['label'].values
    model = None; last_train = -retrain_every

    for t in range(min_train, n):
        if t - last_train >= retrain_every:
            te = t - horizon
            if te < 100: continue
            X = feat[:te]; y = labels[:te]
            m = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            if m.sum() < 50: continue
            model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)
            model.fit(X[m], y[m])
            last_train = t

        if model is not None and not np.isnan(feat[t]).any():
            prob = model.predict_proba(feat[t:t+1])[0]
            probs.iloc[t] = prob[1] if len(prob) > 1 else prob[0]
    return probs

def backtest(df, ml_probs, initial_capital=10000,
             # Sweep these params
             bear_enter_default=0.90, bear_exit_default=1.05,
             bear_enter_ml_low=0.93, bear_exit_ml_high=1.02,
             ml_bear_thresh=0.35, ml_bull_thresh=0.65):
    """v5 仓位逻辑不变，ML 仅调节 regime 切换阈值。"""
    prices = df['price'].values
    sma200 = df['sma200'].values
    rsi10 = df['rsi10'].values
    wret = df['weekly_ret'].values
    dates = df.index

    cash = 0.0
    shares = initial_capital / prices[0]
    in_bear = False
    portfolio = np.zeros(len(dates))

    for i in range(len(dates)):
        price = prices[i]; sma = sma200[i]; rsi = rsi10[i]; wr = wret[i]
        ml_prob = ml_probs.iloc[i] if not np.isnan(ml_probs.iloc[i]) else 0.5

        # ── ML-adjusted regime thresholds ──
        if ml_prob < ml_bear_thresh:
            # ML strongly bearish → enter bear earlier (less confirmation needed)
            be = bear_enter_ml_low
            bx = bear_exit_default
        elif ml_prob > ml_bull_thresh:
            # ML strongly bullish → exit bear earlier
            be = bear_enter_default
            bx = bear_exit_ml_high
        else:
            be = bear_enter_default
            bx = bear_exit_default

        if i >= 200 and not np.isnan(sma):
            if not in_bear and price < sma * be:
                in_bear = True
            elif in_bear and price > sma * bx:
                in_bear = False

        # ── v5 position sizing (unchanged) ──
        if not in_bear:
            if not np.isnan(rsi) and rsi > 80 and not np.isnan(wr) and wr > 0.15:
                target_pct = 0.80
            else:
                target_pct = 1.00
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

def metrics(pv, label=""):
    final = pv.iloc[-1]; start = pv.iloc[0]
    years = (pv.index[-1] - pv.index[0]).days / 365.25
    cagr = (final / start) ** (1/years) - 1
    peak = pv.cummax(); dd = (pv - peak) / peak; mdd = dd.min()
    dr = pv.pct_change().dropna(); rf = 0.045/252
    sharpe = (dr - rf).mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    ds = dr[dr < 0].std()
    sortino = (dr - rf).mean() / ds * np.sqrt(252) if ds > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    return {'label': label, 'final': final, 'cagr': cagr, 'mdd': mdd,
            'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar}

if __name__ == "__main__":
    print("🐻 TQQQ v7.0: v5 + ML Regime Timing")
    data = load_data()
    print(f"Period: {data.index[0].date()} → {data.index[-1].date()}")

    df = compute_features(data['Close'])
    print("Walk-forward ML...")
    ml_probs = walk_forward(df)
    print(f"ML predictions: {ml_probs.notna().sum()} days")

    # v5 baseline
    pv5 = backtest(df, pd.Series(0.5, index=df.index))  # constant 0.5 = no ML effect
    m5 = metrics(pv5, "v5 baseline (no ML)")

    # Buy & Hold
    bh = 10000 / data['Close'].iloc[0] * data['Close']
    mbh = metrics(bh, "Buy & Hold")

    # Grid search ML thresholds
    print("\n🔍 Grid search: ML regime timing parameters...")
    best = None
    results = []

    for be_ml in [0.91, 0.92, 0.93, 0.94, 0.95]:
        for bx_ml in [1.01, 1.02, 1.03]:
            for ml_bt in [0.30, 0.35, 0.40]:
                for ml_bt2 in [0.60, 0.65, 0.70]:
                    pv = backtest(df, ml_probs,
                                  bear_enter_ml_low=be_ml, bear_exit_ml_high=bx_ml,
                                  ml_bear_thresh=ml_bt, ml_bull_thresh=ml_bt2)
                    m = metrics(pv)
                    m['params'] = f"be={be_ml}/bx={bx_ml}/bt={ml_bt}/{ml_bt2}"
                    results.append(m)
                    if best is None or m['calmar'] > best['calmar']:
                        best = m

    print(f"\n🏆 Best ML config: {best['params']}")
    best['label'] = f"v7 ML ({best['params']})"

    # Also test with CAGR as objective
    best_cagr = max(results, key=lambda x: x['cagr'])
    best_cagr['label'] = f"v7 ML maxCAGR ({best_cagr['params']})"

    print(f"\n{'='*80}")
    print(f"{'策略':<45} {'最终价值':>12} {'CAGR':>8} {'MaxDD':>9} {'Sharpe':>8} {'Calmar':>8}")
    print("-" * 95)
    for m in [best, best_cagr, m5, mbh]:
        print(f"{m['label']:<45} ${m['final']:>10,.0f} {m['cagr']*100:>7.1f}% {m['mdd']*100:>8.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f}")
