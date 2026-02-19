#!/usr/bin/env python3
"""
🐻 代码熊 — 个股监控脚本
监控 7 只核心个股的技术面，达到买卖点位时发出警报。
设计为 cron 每小时调用（美股交易时间）。
"""

import json, sys, os
from datetime import datetime, timedelta

# ── 监控标的和触发条件 ──────────────────────────────────────

WATCHLIST = {
    "NVDA": {
        "name": "NVIDIA",
        "buy_zones": [
            {"price": 175, "label": "强支撑位抄底", "priority": "high"},
            {"price": 180, "label": "SMA200附近，可以建仓", "priority": "medium"},
        ],
        "sell_zones": [
            {"price": 210, "label": "接近52周高点，考虑减仓", "priority": "medium"},
            {"price": 250, "label": "接近分析师目标价$254，止盈", "priority": "high"},
        ],
        "alert_rsi_low": 30,
        "alert_rsi_high": 80,
        "notes": "2/25财报催化剂",
    },
    "PLTR": {
        "name": "Palantir",
        "buy_zones": [
            {"price": 127, "label": "2月低点支撑，强买点", "priority": "high"},
            {"price": 115, "label": "极端恐慌价，重仓", "priority": "high"},
            {"price": 133, "label": "当前价位可分批建仓", "priority": "low"},
        ],
        "sell_zones": [
            {"price": 170, "label": "分析师目标区间下沿，考虑减仓", "priority": "medium"},
            {"price": 190, "label": "分析师目标$190，止盈", "priority": "high"},
        ],
        "alert_rsi_low": 28,
        "alert_rsi_high": 75,
        "stop_loss": 120,
        "notes": "RSI<32超卖反弹机会",
    },
    "TSLA": {
        "name": "Tesla",
        "buy_zones": [
            {"price": 374, "label": "强支撑位，如果到这里可以入", "priority": "high"},
            {"price": 400, "label": "$400心理关口确认后可买", "priority": "medium"},
        ],
        "sell_zones": [
            {"price": 467, "label": "接近阻力位，减仓", "priority": "medium"},
            {"price": 500, "label": "接近52周高点$499，止盈", "priority": "high"},
        ],
        "alert_rsi_low": 30,
        "alert_rsi_high": 80,
        "stop_loss": 350,
        "notes": "期权情绪极空(87% put)，等企稳",
    },
    "RKLB": {
        "name": "Rocket Lab",
        "buy_zones": [
            {"price": 55, "label": "极端低位，长线好买点", "priority": "high"},
            {"price": 60, "label": "心理关口支撑", "priority": "medium"},
        ],
        "sell_zones": [
            {"price": 80, "label": "第二阻力位，部分止盈", "priority": "medium"},
            {"price": 90, "label": "前高附近，止盈", "priority": "high"},
        ],
        "alert_rsi_low": 28,
        "alert_rsi_high": 75,
        "notes": "2/26财报，Neutron进展是关键",
    },
    "VRT": {
        "name": "Vertiv",
        "buy_zones": [
            {"price": 220, "label": "回调至SMA20附近，理想买点", "priority": "high"},
            {"price": 230, "label": "小幅回调，可以开始建仓", "priority": "medium"},
            {"price": 200, "label": "SMA50附近，重仓买入", "priority": "high"},
        ],
        "sell_zones": [
            {"price": 279, "label": "Seeking Alpha目标$279，减仓", "priority": "medium"},
            {"price": 300, "label": "整数关口，止盈", "priority": "high"},
        ],
        "alert_rsi_low": 35,
        "alert_rsi_high": 80,
        "notes": "刚涨20%，等回调再入",
    },
    "APP": {
        "name": "AppLovin",
        "buy_zones": [
            {"price": 350, "label": "加仓区域", "priority": "high"},
            {"price": 376, "label": "当前价位可建仓（深度超卖）", "priority": "medium"},
            {"price": 300, "label": "极端恐慌价，止损线附近", "priority": "high"},
        ],
        "sell_zones": [
            {"price": 500, "label": "修复过半，部分止盈", "priority": "medium"},
            {"price": 600, "label": "接近前高，大幅止盈", "priority": "high"},
            {"price": 650, "label": "分析师目标$652，清仓", "priority": "high"},
        ],
        "alert_rsi_low": 25,
        "alert_rsi_high": 75,
        "stop_loss": 280,
        "notes": "基本面与技术面严重背离，反转机会",
    },
    "CLS": {
        "name": "Celestica",
        "buy_zones": [
            {"price": 270, "label": "200日线附近，好买点", "priority": "high"},
            {"price": 250, "label": "深度回调，重仓", "priority": "high"},
            {"price": 280, "label": "当前价位可小仓位建仓", "priority": "low"},
        ],
        "sell_zones": [
            {"price": 350, "label": "目标区间下沿", "priority": "medium"},
            {"price": 391, "label": "Barclays目标$391，止盈", "priority": "high"},
        ],
        "alert_rsi_low": 30,
        "alert_rsi_high": 78,
        "stop_loss": 250,
        "notes": "跌破200日线是黄旗，但基本面强",
    },
}


# ── 数据获取 ─────────────────────────────────────────────────

def fetch_prices(tickers):
    """Fetch current prices and basic technicals via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("❌ yfinance not installed", file=sys.stderr)
        return None

    results = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            if hist.empty:
                continue

            close = hist['Close']
            price = close.iloc[-1]

            # SMA
            sma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None
            sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
            sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

            # RSI(14)
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, float('nan'))
            rsi = (100 - (100 / (1 + rs))).iloc[-1]

            # Weekly return
            weekly_ret = (price - close.iloc[-6]) / close.iloc[-6] if len(close) >= 6 else 0

            # Daily return
            daily_ret = (price - close.iloc[-2]) / close.iloc[-2] if len(close) >= 2 else 0

            # 52-week high/low
            high_52w = close.max()
            low_52w = close.min()

            # ATH drawdown
            dd_from_high = (price - high_52w) / high_52w * 100

            results[ticker] = {
                "price": round(float(price), 2),
                "sma20": round(float(sma20), 2) if sma20 else None,
                "sma50": round(float(sma50), 2) if sma50 else None,
                "sma200": round(float(sma200), 2) if sma200 else None,
                "rsi14": round(float(rsi), 1) if rsi == rsi else None,
                "weekly_ret": round(float(weekly_ret) * 100, 1),
                "daily_ret": round(float(daily_ret) * 100, 1),
                "high_52w": round(float(high_52w), 2),
                "low_52w": round(float(low_52w), 2),
                "dd_from_high": round(float(dd_from_high), 1),
            }
        except Exception as e:
            print(f"⚠️ {ticker}: {e}", file=sys.stderr)
            continue

    return results


# ── 信号检测 ─────────────────────────────────────────────────

def check_signals(prices):
    """Check all watchlist stocks against trigger levels."""
    alerts = []

    for ticker, config in WATCHLIST.items():
        if ticker not in prices:
            continue

        data = prices[ticker]
        price = data["price"]
        rsi = data.get("rsi14")
        name = config["name"]

        # Check buy zones
        for zone in config["buy_zones"]:
            if price <= zone["price"]:
                alerts.append({
                    "ticker": ticker,
                    "name": name,
                    "type": "BUY",
                    "price": price,
                    "trigger": zone["price"],
                    "label": zone["label"],
                    "priority": zone["priority"],
                    "rsi": rsi,
                })

        # Check sell zones
        for zone in config["sell_zones"]:
            if price >= zone["price"]:
                alerts.append({
                    "ticker": ticker,
                    "name": name,
                    "type": "SELL",
                    "price": price,
                    "trigger": zone["price"],
                    "label": zone["label"],
                    "priority": zone["priority"],
                    "rsi": rsi,
                })

        # Check RSI extremes
        if rsi is not None:
            if rsi <= config.get("alert_rsi_low", 30):
                alerts.append({
                    "ticker": ticker,
                    "name": name,
                    "type": "RSI_OVERSOLD",
                    "price": price,
                    "trigger": config["alert_rsi_low"],
                    "label": f"RSI={rsi:.1f} 超卖，关注抄底机会",
                    "priority": "medium",
                    "rsi": rsi,
                })
            elif rsi >= config.get("alert_rsi_high", 80):
                alerts.append({
                    "ticker": ticker,
                    "name": name,
                    "type": "RSI_OVERBOUGHT",
                    "price": price,
                    "trigger": config["alert_rsi_high"],
                    "label": f"RSI={rsi:.1f} 超买，注意风险",
                    "priority": "medium",
                    "rsi": rsi,
                })

        # Check stop loss
        if "stop_loss" in config and price <= config["stop_loss"]:
            alerts.append({
                "ticker": ticker,
                "name": name,
                "type": "STOP_LOSS",
                "price": price,
                "trigger": config["stop_loss"],
                "label": f"跌破止损位 ${config['stop_loss']}！",
                "priority": "critical",
                "rsi": rsi,
            })

    return alerts


# ── 格式化输出 ────────────────────────────────────────────────

def format_dashboard(prices):
    """Format a quick dashboard view."""
    lines = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"🐻 个股监控 — {now}")
    lines.append("=" * 55)
    lines.append(f"{'股票':<6} {'价格':>8} {'日涨跌':>7} {'周涨跌':>7} {'RSI':>6} {'vs高点':>7}")
    lines.append("-" * 55)

    for ticker in WATCHLIST:
        if ticker not in prices:
            lines.append(f"{ticker:<6} {'N/A':>8}")
            continue
        d = prices[ticker]
        rsi_str = f"{d['rsi14']:.0f}" if d['rsi14'] else "N/A"
        lines.append(
            f"{ticker:<6} ${d['price']:>7.2f} {d['daily_ret']:>+6.1f}% {d['weekly_ret']:>+6.1f}% "
            f"{rsi_str:>5} {d['dd_from_high']:>+6.1f}%"
        )

    return "\n".join(lines)


def format_alerts(alerts):
    """Format alert messages."""
    if not alerts:
        return ""

    # Sort by priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    alerts.sort(key=lambda a: priority_order.get(a["priority"], 9))

    lines = []
    lines.append("")
    lines.append("⚡ 触发警报:")

    for a in alerts:
        emoji = {"BUY": "🟢", "SELL": "🔴", "RSI_OVERSOLD": "📉",
                 "RSI_OVERBOUGHT": "📈", "STOP_LOSS": "🚨"}.get(a["type"], "⚠️")
        prio = {"critical": "‼️", "high": "❗", "medium": "⚠️", "low": "ℹ️"}.get(a["priority"], "")
        lines.append(f"  {emoji} {prio} {a['ticker']} ${a['price']:.2f} — {a['label']}")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Stock Monitor')
    parser.add_argument('--json', action='store_true', help='JSON output')
    parser.add_argument('--alerts-only', action='store_true', help='Only output if alerts')
    parser.add_argument('--dashboard', action='store_true', help='Show dashboard even without alerts')
    args = parser.parse_args()

    tickers = list(WATCHLIST.keys())
    prices = fetch_prices(tickers)

    if not prices:
        print("❌ 无法获取数据", file=sys.stderr)
        sys.exit(1)

    alerts = check_signals(prices)

    if args.json:
        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "prices": prices,
            "alerts": alerts,
            "has_alerts": len(alerts) > 0,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    elif args.alerts_only and not alerts:
        # No alerts, silent exit
        sys.exit(0)
    else:
        print(format_dashboard(prices))
        alert_text = format_alerts(alerts)
        if alert_text:
            print(alert_text)
        elif not args.dashboard:
            print("\n✅ 无异常信号")
