#!/usr/bin/env python3
"""
🐻 TQQQ Beast v8 — Daily Monitor
每日拉最新数据，输出当前信号和买卖提醒。
设计为 cron 调用，输出纯文本摘要。
"""

import pandas as pd
import numpy as np
import os, sys, json
from datetime import datetime, timedelta

# ── Data fetching ──────────────────────────────────────────────

def fetch_latest_data():
    """Try stooq CSV update, then yfinance, then use cached CSV."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tqqq_daily.csv')
    
    # Try yfinance for fresh data
    try:
        import yfinance as yf
        df = yf.download('TQQQ', start='2010-01-01', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df[['Close']].dropna()
        if len(df) > 3000:
            # Save updated CSV
            df.to_csv(csv_path)
            return df
    except Exception as e:
        pass

    # Fallback to cached CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')[['Close']].dropna()
        df.sort_index(inplace=True)
        return df
    
    raise RuntimeError("No data source available")


# ── Indicators (same as v8) ────────────────────────────────────

def compute_sma(prices, window):
    return prices.rolling(window).mean()

def compute_rsi(prices, period=10):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_realized_vol(prices, window=20):
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)

def sigmoid(x, center=30, steepness=-0.20):
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))


# ── Strategy Parameters (v8) ──────────────────────────────────

BEAR_BAND = 0.93
BULL_BAND = 1.05
VOL_THRESHOLD = 0.65
BEAR_FLOOR = 0.25
BEAR_CEILING = 0.95
RSI_CENTER = 30


# ── Signal Generation ─────────────────────────────────────────

def generate_signal(df):
    """Generate current signal from price data."""
    prices = df['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    rsi14 = compute_rsi(prices, 14)
    weekly_ret = prices.pct_change(5)
    vol = compute_realized_vol(prices, 20)
    
    # Determine current regime by replaying history
    in_bear = False
    for i in range(200, len(prices)):
        sma = sma200.iloc[i]
        price = prices.iloc[i]
        if pd.isna(sma):
            continue
        if not in_bear and price < sma * BEAR_BAND:
            in_bear = True
        elif in_bear and price > sma * BULL_BAND:
            in_bear = False
    
    # Current values
    last = len(prices) - 1
    price = prices.iloc[last]
    sma = sma200.iloc[last]
    rsi = rsi10.iloc[last]
    rsi_14 = rsi14.iloc[last]
    wret = weekly_ret.iloc[last]
    v = vol.iloc[last]
    date = prices.index[last]
    
    # Band triggers
    bear_trigger = sma * BEAR_BAND
    bull_trigger = sma * BULL_BAND
    
    # Recommended position
    if not in_bear:
        if not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15:
            position = 0.80
            pos_reason = "牛市狂热减仓"
        elif not pd.isna(v) and v > VOL_THRESHOLD:
            position = 0.85
            pos_reason = f"牛市但波动率偏高 ({v*100:.0f}%)"
        else:
            position = 1.00
            pos_reason = "牛市全仓"
    else:
        if not pd.isna(rsi):
            sig_val = sigmoid(rsi, RSI_CENTER, -0.20)
            position = BEAR_FLOOR + (BEAR_CEILING - BEAR_FLOOR) * sig_val
            if not pd.isna(wret) and wret < -0.12:
                position = max(position, 0.80)
                pos_reason = f"熊市恐慌抄底 (周跌{wret*100:.1f}%)"
            else:
                pos_reason = f"熊市 sigmoid 仓位 (RSI={rsi:.0f})"
        else:
            position = BEAR_FLOOR
            pos_reason = "熊市底仓"
    
    # Alerts
    alerts = []
    pct_to_bear = (price - bear_trigger) / price * 100
    pct_to_bull = (bull_trigger - price) / price * 100
    
    if not in_bear and pct_to_bear < 5:
        alerts.append(f"⚠️ 距熊市触发仅 {pct_to_bear:.1f}%（{bear_trigger:.2f}），注意减仓")
    if in_bear and pct_to_bull < 5:
        alerts.append(f"🟢 距牛市确认仅 {pct_to_bull:.1f}%（{bull_trigger:.2f}），准备加仓")
    if not pd.isna(rsi) and rsi < 25:
        alerts.append(f"📉 RSI(10) = {rsi:.1f}，极度超卖，考虑加仓")
    if not pd.isna(rsi) and rsi > 80:
        alerts.append(f"📈 RSI(10) = {rsi:.1f}，极度超买，考虑减仓")
    if not pd.isna(v) and v > 0.80:
        alerts.append(f"🌪️ 波动率 {v*100:.0f}% 极高，风险警告")
    if not pd.isna(wret) and wret < -0.10:
        alerts.append(f"💥 本周跌 {wret*100:.1f}%，可能触发抄底信号")
    
    # 20-day price change
    if last >= 20:
        pct_20d = (price - prices.iloc[last-20]) / prices.iloc[last-20] * 100
    else:
        pct_20d = 0
    
    return {
        'date': date,
        'price': price,
        'sma200': sma,
        'rsi10': rsi,
        'rsi14': rsi_14,
        'weekly_ret': wret,
        'realized_vol': v,
        'regime': 'BEAR' if in_bear else 'BULL',
        'position': position,
        'pos_reason': pos_reason,
        'bear_trigger': bear_trigger,
        'bull_trigger': bull_trigger,
        'pct_to_bear': pct_to_bear,
        'pct_to_bull': pct_to_bull,
        'pct_20d': pct_20d,
        'alerts': alerts,
    }


def format_signal(sig):
    """Format signal as readable text."""
    lines = []
    lines.append(f"🐻 TQQQ Beast v8 Monitor — {sig['date'].strftime('%Y-%m-%d')}")
    lines.append("=" * 45)
    lines.append(f"💰 TQQQ: ${sig['price']:.2f}")
    lines.append(f"📊 SMA200: ${sig['sma200']:.2f}")
    lines.append(f"📉 RSI(10): {sig['rsi10']:.1f}")
    lines.append(f"📈 RSI(14): {sig['rsi14']:.1f}")
    lines.append(f"📅 周涨跌: {sig['weekly_ret']*100:+.1f}%")
    lines.append(f"🌪️ 波动率: {sig['realized_vol']*100:.0f}% (年化)")
    lines.append(f"📊 20日涨跌: {sig['pct_20d']:+.1f}%")
    lines.append("")
    
    regime_emoji = "🐻" if sig['regime'] == 'BEAR' else "🐂"
    lines.append(f"{regime_emoji} Regime: {sig['regime']}")
    lines.append(f"🎯 推荐仓位: {sig['position']*100:.0f}% — {sig['pos_reason']}")
    lines.append("")
    
    if sig['regime'] == 'BULL':
        lines.append(f"🔴 熊市触发: ${sig['bear_trigger']:.2f} (距离 {sig['pct_to_bear']:.1f}%)")
    else:
        lines.append(f"🟢 牛市确认: ${sig['bull_trigger']:.2f} (需涨 {sig['pct_to_bull']:.1f}%)")
        lines.append(f"🔴 当前熊市底仓: {sig['position']*100:.0f}%")
    
    if sig['alerts']:
        lines.append("")
        lines.append("⚡ 警报:")
        for a in sig['alerts']:
            lines.append(f"  {a}")
    else:
        lines.append("")
        lines.append("✅ 无异常信号")
    
    return "\n".join(lines)


def has_alerts(sig):
    """Check if there are actionable alerts."""
    return len(sig['alerts']) > 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='TQQQ Beast v8 Daily Monitor')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--alerts-only', action='store_true', help='Only output if alerts exist')
    args = parser.parse_args()
    
    try:
        df = fetch_latest_data()
        sig = generate_signal(df)
        
        if args.alerts_only and not has_alerts(sig):
            sys.exit(0)
        
        if args.json:
            output = {k: (v.isoformat() if isinstance(v, (datetime, pd.Timestamp)) else 
                         float(v) if isinstance(v, (np.floating, np.integer)) else v)
                      for k, v in sig.items()}
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            print(format_signal(sig))
    except Exception as e:
        print(f"❌ Monitor error: {e}", file=sys.stderr)
        sys.exit(1)
