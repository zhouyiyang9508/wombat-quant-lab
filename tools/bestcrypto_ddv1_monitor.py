#!/usr/bin/env python3
"""
BestCrypto DDv1 Daily Monitor
代码熊 🐻 | 2026-02-20

策略：
- Crypto 部分：BestCrypto（BTC/ETH/GLD 90日动量轮动）
- Stock 部分：QQQ（代替 Stock v3b 选股）
- 组合方式：DDv1（逆波动率 + 回撤减权 + 现金仓位）

监控逻辑：
1. 每天计算当前 regime（BTC/ETH/GLD 谁最强）
2. 计算 DDv1 权重（crypto vs QQQ）
3. 如果 regime 变化或权重变化 >5%，触发调仓通知

使用：python3 bestcrypto_ddv1_monitor.py [--dashboard] [--history N]
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data_cache"
STATE_FILE = BASE / "tools" / "bestcrypto_ddv1_state.json"


def download_latest():
    """Download latest price data via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("⚠️ yfinance not installed, using cached data")
        return False

    tickers = {
        "BTC-USD": "BTC_USD.csv",
        "ETH-USD": "ETH_USD.csv",
        "GLD": "GLD.csv",
        "QQQ": "QQQ.csv",
    }

    for symbol, filename in tickers.items():
        try:
            df = yf.download(symbol, start="2015-01-01", progress=False, auto_adjust=True)
            if len(df) < 10:
                print(f"  ⚠️ {symbol}: only {len(df)} rows, skipping")
                continue
            df = df.reset_index()
            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df.to_csv(DATA_DIR / filename, index=False)
        except Exception as e:
            print(f"  ⚠️ {symbol} download failed: {e}")
    return True


def load_prices():
    """Load price data for all assets."""
    assets = {}
    for name, filename in [("BTC", "BTC_USD.csv"), ("ETH", "ETH_USD.csv"),
                            ("GLD", "GLD.csv"), ("QQQ", "QQQ.csv")]:
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run with --update first.")
        df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
        assets[name] = df["Close"].dropna()
    return assets


def calc_momentum(prices, lookback=90):
    """Calculate momentum using yesterday's close (no look-ahead)."""
    shifted = prices.shift(1)  # yesterday's close
    shifted_lb = prices.shift(1 + lookback)  # lookback days before yesterday
    return (shifted / shifted_lb - 1).dropna()


def get_bestcrypto_regime(btc_mom, eth_mom, gld_mom):
    """Determine BestCrypto regime."""
    moms = {"BTC": btc_mom, "ETH": eth_mom, "GLD": gld_mom}
    best = max(moms, key=moms.get)

    if best == "BTC" and btc_mom > 0:
        crypto_w = {"BTC": 0.70, "ETH": 0.15, "GLD": 0.10}
        regime = "🟢 BTC Bull"
    elif best == "ETH" and eth_mom > 0:
        crypto_w = {"BTC": 0.15, "ETH": 0.65, "GLD": 0.10}
        regime = "🟣 ETH Bull"
    elif best == "GLD":
        crypto_w = {"BTC": 0.15, "ETH": 0.10, "GLD": 0.55}
        regime = "🟡 GLD Defense"
    else:
        crypto_w = {"BTC": 0.15, "ETH": 0.10, "GLD": 0.35}
        regime = "🔴 All Weak (Cash)"

    cash_pct = 1.0 - sum(crypto_w.values())
    if cash_pct > 0.01:
        crypto_w["CASH"] = round(cash_pct, 2)

    return regime, best, crypto_w


def calc_ddv1_weights(crypto_returns, qqq_returns, lookback=20, dd_threshold=-0.10):
    """Calculate DDv1 portfolio weights (crypto vs QQQ)."""
    n = len(crypto_returns)
    if n < lookback:
        return 0.5, 0.5, 0.0, 0.0

    # Rolling volatility
    crypto_vol = crypto_returns.iloc[-lookback:].std() * np.sqrt(252)
    qqq_vol = qqq_returns.iloc[-lookback:].std() * np.sqrt(252)

    crypto_vol = max(crypto_vol, 0.01)
    qqq_vol = max(qqq_vol, 0.01)

    # Inverse vol weights
    inv_c = 1.0 / crypto_vol
    inv_q = 1.0 / qqq_vol
    total = inv_c + inv_q
    w_crypto = inv_c / total
    w_qqq = inv_q / total

    # Current drawdown of each
    crypto_cum = (1 + crypto_returns).cumprod()
    qqq_cum = (1 + qqq_returns).cumprod()

    crypto_peak = crypto_cum.max()
    qqq_peak = qqq_cum.max()

    crypto_dd = (crypto_cum.iloc[-1] - crypto_peak) / crypto_peak
    qqq_dd = (qqq_cum.iloc[-1] - qqq_peak) / qqq_peak

    # DD reduction
    if crypto_dd < dd_threshold:
        reduction = max(0.3, 1.0 + crypto_dd * 2)
        w_crypto *= reduction
    if qqq_dd < dd_threshold:
        reduction = max(0.3, 1.0 + qqq_dd * 2)
        w_qqq *= reduction

    cash = 1.0 - w_crypto - w_qqq

    return w_crypto, w_qqq, crypto_dd, qqq_dd


def calc_final_positions(crypto_weights, w_crypto, w_qqq):
    """Calculate final dollar positions for $10,000 portfolio."""
    positions = {}

    # Crypto portion
    for asset, pct in crypto_weights.items():
        if asset == "CASH":
            positions["CASH (crypto)"] = w_crypto * pct
        else:
            positions[asset] = w_crypto * pct

    # QQQ portion
    positions["QQQ"] = w_qqq

    # DDv1 cash
    ddv1_cash = 1.0 - w_crypto - w_qqq
    if ddv1_cash > 0.01:
        positions["CASH (DDv1)"] = ddv1_cash

    return positions


def load_state():
    """Load previous state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    """Save current state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def check_alerts(prev_state, regime, best_asset, w_crypto, w_qqq, positions):
    """Check if rebalancing is needed."""
    alerts = []

    prev_regime = prev_state.get("regime", "")
    prev_best = prev_state.get("best_asset", "")
    prev_w_crypto = prev_state.get("w_crypto", 0.5)
    prev_w_qqq = prev_state.get("w_qqq", 0.5)

    # Regime change
    if prev_best and prev_best != best_asset:
        alerts.append(f"🔄 **REGIME CHANGE**: {prev_regime} → {regime}")
        alerts.append(f"   动量领先从 {prev_best} 切换到 {best_asset}")
        alerts.append(f"   ➡️ **需要调仓 crypto 部分**")

    # Weight change > 5%
    crypto_diff = abs(w_crypto - prev_w_crypto)
    qqq_diff = abs(w_qqq - prev_w_qqq)
    if crypto_diff > 0.05 or qqq_diff > 0.05:
        alerts.append(f"⚖️ **DDv1 权重变化**: Crypto {prev_w_crypto*100:.0f}%→{w_crypto*100:.0f}%, QQQ {prev_w_qqq*100:.0f}%→{w_qqq*100:.0f}%")
        alerts.append(f"   ➡️ **需要调整 crypto/QQQ 比例**")

    return alerts


def format_dashboard(date, prices, moms, regime, best_asset, crypto_weights,
                     w_crypto, w_qqq, crypto_dd, qqq_dd, positions, alerts):
    """Format dashboard output."""
    lines = []
    lines.append(f"📊 **BestCrypto DDv1 Monitor** — {date}")
    lines.append("")

    # Current prices
    lines.append("**💰 当前价格**")
    for asset, price in prices.items():
        lines.append(f"  {asset}: ${price:,.2f}")
    lines.append("")

    # 90-day momentum
    lines.append("**📈 90日动量**")
    for asset, mom in moms.items():
        emoji = "🟢" if mom > 0 else "🔴"
        lines.append(f"  {emoji} {asset}: {mom*100:+.1f}%")
    lines.append("")

    # Regime
    lines.append(f"**🎯 当前 Regime**: {regime}")
    lines.append(f"  Crypto 内部配置: {' / '.join(f'{k} {v*100:.0f}%' for k, v in crypto_weights.items())}")
    lines.append("")

    # DDv1 weights
    cash_pct = max(0, 1.0 - w_crypto - w_qqq)
    lines.append(f"**⚖️ DDv1 组合权重**")
    lines.append(f"  Crypto: {w_crypto*100:.1f}% | QQQ: {w_qqq*100:.1f}% | Cash: {cash_pct*100:.1f}%")
    lines.append(f"  Crypto DD: {crypto_dd*100:.1f}% | QQQ DD: {qqq_dd*100:.1f}%")
    lines.append("")

    # Final positions
    lines.append(f"**💼 最终持仓（$10,000 本金）**")
    for asset, pct in sorted(positions.items(), key=lambda x: -x[1]):
        dollar = pct * 10000
        lines.append(f"  {asset}: {pct*100:.1f}% (${dollar:,.0f})")
    lines.append("")

    # Alerts
    if alerts:
        lines.append("**🚨 调仓提醒**")
        for alert in alerts:
            lines.append(f"  {alert}")
    else:
        lines.append("✅ **无需调仓** — 持仓不变")

    return "\n".join(lines)


def main():
    args = sys.argv[1:]
    update = "--update" in args or "--dashboard" in args

    # Download latest data
    if update:
        print("📡 Updating price data...")
        download_latest()

    # Load prices
    assets = load_prices()
    btc = assets["BTC"]
    eth = assets["ETH"]
    gld = assets["GLD"]
    qqq = assets["QQQ"]

    # Latest date (common)
    common_end = min(btc.index[-1], eth.index[-1], gld.index[-1], qqq.index[-1])
    today = common_end.strftime("%Y-%m-%d")

    # Current prices
    current_prices = {
        "BTC": btc.loc[:common_end].iloc[-1],
        "ETH": eth.loc[:common_end].iloc[-1],
        "GLD": gld.loc[:common_end].iloc[-1],
        "QQQ": qqq.loc[:common_end].iloc[-1],
    }

    # 90-day momentum (using yesterday's close)
    btc_mom = calc_momentum(btc, 90)
    eth_mom = calc_momentum(eth, 90)
    gld_mom = calc_momentum(gld, 90)

    latest_moms = {
        "BTC": btc_mom.loc[:common_end].iloc[-1],
        "ETH": eth_mom.loc[:common_end].iloc[-1],
        "GLD": gld_mom.loc[:common_end].iloc[-1],
    }

    # BestCrypto regime
    regime, best_asset, crypto_weights = get_bestcrypto_regime(
        latest_moms["BTC"], latest_moms["ETH"], latest_moms["GLD"]
    )

    # Simulate BestCrypto daily returns for DDv1 calculation
    # Align all on common trading days
    common_idx = btc.index.intersection(eth.index).intersection(gld.index).intersection(qqq.index)
    common_idx = common_idx.sort_values()

    btc_r = btc.loc[common_idx].pct_change()
    eth_r = eth.loc[common_idx].pct_change()
    gld_r = gld.loc[common_idx].pct_change()
    qqq_r = qqq.loc[common_idx].pct_change()

    # BestCrypto composite returns (simplified: use regime weights)
    # For DDv1, we need historical crypto returns
    btc_mom_series = calc_momentum(btc.loc[common_idx], 90)
    eth_mom_series = calc_momentum(eth.loc[common_idx], 90)
    gld_mom_series = calc_momentum(gld.loc[common_idx], 90)

    # Build crypto return series
    crypto_r = pd.Series(0.0, index=common_idx)
    for i in range(1, len(common_idx)):
        dt = common_idx[i]
        if dt in btc_mom_series.index and dt in eth_mom_series.index and dt in gld_mom_series.index:
            _, _, cw = get_bestcrypto_regime(
                btc_mom_series.loc[dt], eth_mom_series.loc[dt], gld_mom_series.loc[dt]
            )
            cr = 0.0
            for asset, w in cw.items():
                if asset == "BTC":
                    cr += w * btc_r.loc[dt] if not pd.isna(btc_r.loc[dt]) else 0
                elif asset == "ETH":
                    cr += w * eth_r.loc[dt] if not pd.isna(eth_r.loc[dt]) else 0
                elif asset == "GLD":
                    cr += w * gld_r.loc[dt] if not pd.isna(gld_r.loc[dt]) else 0
            crypto_r.loc[dt] = cr

    crypto_r = crypto_r.dropna()
    qqq_r_clean = qqq_r.dropna()

    # Align
    common_r = crypto_r.index.intersection(qqq_r_clean.index)
    crypto_r = crypto_r.loc[common_r]
    qqq_r_aligned = qqq_r_clean.loc[common_r]

    # DDv1 weights
    w_crypto, w_qqq, crypto_dd, qqq_dd = calc_ddv1_weights(crypto_r, qqq_r_aligned)

    # Final positions
    positions = calc_final_positions(crypto_weights, w_crypto, w_qqq)

    # Load previous state and check alerts
    prev_state = load_state()
    alerts = check_alerts(prev_state, regime, best_asset, w_crypto, w_qqq, positions)

    # Save current state
    new_state = {
        "date": today,
        "regime": regime,
        "best_asset": best_asset,
        "w_crypto": round(w_crypto, 4),
        "w_qqq": round(w_qqq, 4),
        "crypto_dd": round(crypto_dd, 4),
        "qqq_dd": round(qqq_dd, 4),
        "positions": {k: round(v, 4) for k, v in positions.items()},
        "prices": {k: round(v, 2) for k, v in current_prices.items()},
        "momentums": {k: round(v, 4) for k, v in latest_moms.items()},
    }
    save_state(new_state)

    # Output
    dashboard = format_dashboard(
        today, current_prices, latest_moms, regime, best_asset,
        crypto_weights, w_crypto, w_qqq, crypto_dd, qqq_dd, positions, alerts
    )
    print(dashboard)

    # Return alerts for cron
    if alerts:
        return 1  # has alerts
    return 0


if __name__ == "__main__":
    sys.exit(main())
