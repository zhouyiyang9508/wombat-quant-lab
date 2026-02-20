#!/usr/bin/env python3
"""Calculate current positions for all 5 strategies as of 2026-02-20 using cached data."""

import json, os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

TODAY = "2026-02-20"
SIX_MONTHS_AGO = "2025-08-20"
CACHE = "data_cache"

def load_csv(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.strip() for c in df.columns]
    # Use 'Close' or last column
    if 'Close' in df.columns:
        return df['Close']
    return df.iloc[:, 3]  # 4th col is usually close

def get_price(name, filename=None):
    fn = filename or f"{name}.csv"
    return load_csv(os.path.join(CACHE, fn))

def momentum(series, ref_date=SIX_MONTHS_AGO):
    s = series.dropna()
    current = s.iloc[-1]
    past = s[s.index <= ref_date]
    if len(past) == 0:
        past_price = s.iloc[0]
    else:
        past_price = past.iloc[-1]
    return (current / past_price) - 1, current

# ============ CORE DATA ============
btc = get_price("BTC_USD")
gld = get_price("GLD")
qqq = get_price("QQQ")
spy = get_price("SPY")
tqqq = get_price("TQQQ")

print(f"Data up to: BTC {btc.index[-1].date()}, SPY {spy.index[-1].date()}")

# ============ STRATEGY 1: BTC v7f DualMom ============
btc_mom, btc_price = momentum(btc)
gld_mom, gld_price = momentum(gld)
dual_mom_hold = "BTC" if btc_mom > gld_mom else "GLD"

# ============ STRATEGY 5: TQQQ v9g GLD ============
qqq_s = qqq.dropna()
qqq_current = qqq_s.iloc[-1]
sma200 = qqq_s.rolling(200).mean().iloc[-1]
tqqq_state = "牛市" if qqq_current > sma200 else "熊市"
tqqq_config = "100% TQQQ" if tqqq_state == "牛市" else "30% TQQQ + 70% GLD"

# ============ STOCKS ============
with open(f'{CACHE}/sp500_sectors.json') as f:
    sector_map = json.load(f)

stock_data = {}
for fn in os.listdir(f'{CACHE}/stocks/'):
    if not fn.endswith('.csv'):
        continue
    ticker = fn[:-4]
    if ticker not in sector_map:
        continue
    try:
        s = load_csv(f'{CACHE}/stocks/{fn}')
        s = s.dropna()
        if len(s) < 60:
            continue
        mom, price = momentum(s)
        stock_data[ticker] = {'momentum': mom, 'price': price, 'sector': sector_map[ticker]}
    except:
        pass

print(f"Loaded {len(stock_data)} stocks")

# Sector momentum
sector_moms = defaultdict(list)
for t, info in stock_data.items():
    sector_moms[info['sector']].append(info['momentum'])

sector_avg = {s: np.mean(moms) for s, moms in sector_moms.items() if len(moms) >= 3}
top3_sectors = sorted(sector_avg.items(), key=lambda x: x[1], reverse=True)[:3]

# Top 3 stocks per sector
positions = []
for sector, sec_mom in top3_sectors:
    sector_stocks = [(t, info) for t, info in stock_data.items() if info['sector'] == sector]
    sector_stocks.sort(key=lambda x: x[1]['momentum'], reverse=True)
    for t, info in sector_stocks[:3]:
        positions.append((t, sector, info['momentum'], info['price']))

weight_per_stock = 100.0 / len(positions) if positions else 0

# SPY drawdown
spy_s = spy.dropna()
spy_peak = spy_s.max()
spy_dd = (spy_s.iloc[-1] / spy_peak) - 1
gld_hedge_triggered = spy_dd < -0.08

# Portfolio DD
portfolio_dd = spy_dd
dd_triggered = portfolio_dd < -0.15

# ============ OUTPUT ============
btc_alloc = 0.4 if dd_triggered else 0.6
stock_alloc = 0.6 if dd_triggered else 0.4

output = f"""# 当前持仓报告 - {TODAY}

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
> 数据截至: {spy.index[-1].strftime('%Y-%m-%d')}（最近交易日）

---

## 策略 1: BTC v7f DualMom

**当前持仓: 100% {dual_mom_hold}**

| 指标 | 值 |
|------|------|
| BTC 6M 动量 | {btc_mom*100:+.1f}% |
| GLD 6M 动量 | {gld_mom*100:+.1f}% |
| BTC 价格 | ${btc_price:,.0f} |
| GLD 价格 | ${gld_price:.2f} |

---

## 策略 2: Stock v4d GLD Hedge

**Top 3 行业:**
"""
for i, (sector, mom) in enumerate(top3_sectors, 1):
    output += f"{i}. {sector} (动量 {mom*100:+.1f}%)\n"

if gld_hedge_triggered:
    output += f"\n**持仓明细（SPY 回撤 {spy_dd*100:.1f}%，触发 GLD 对冲）:**\n- 70% Stock + 30% GLD\n\nStock 部分:\n"
else:
    output += f"\n**持仓明细（SPY 回撤 {spy_dd*100:.1f}%，未触发 GLD 对冲）:**\n"

for i, (ticker, sector, mom, price) in enumerate(positions, 1):
    w = weight_per_stock * (0.7 if gld_hedge_triggered else 1.0)
    output += f"{i}. {ticker} ({sector}) - ${price:.2f} - 动量 {mom*100:+.1f}% - 权重 {w:.1f}%\n"

if gld_hedge_triggered:
    output += f"\n10. GLD - ${gld_price:.2f} - 权重 30.0%\n"

output += f"""
---

## 策略 3: Stock v3b

**Top 3 行业:** 同 v4d

**持仓明细（无 GLD 对冲）:**
"""
for i, (ticker, sector, mom, price) in enumerate(positions, 1):
    output += f"{i}. {ticker} ({sector}) - ${price:.2f} - 动量 {mom*100:+.1f}% - 权重 {weight_per_stock:.1f}%\n"

output += f"""
---

## 策略 4: Portfolio DD Responsive

**组合回撤: {portfolio_dd*100:.1f}% ({'触发调整' if dd_triggered else '正常'})**

"""
if dd_triggered:
    output += f"**配置（回撤触发调整）:**\n- 40% BTC v7f → 100% {dual_mom_hold}\n- 60% Stock v3b → 9 只股票\n"
else:
    output += f"**配置（正常）:**\n- 60% BTC v7f → 100% {dual_mom_hold}\n- 40% Stock v3b → 9 只股票\n"

output += "\n**实际持仓:**\n"
if dual_mom_hold == "BTC":
    output += f"- BTC: {btc_alloc*100:.0f}%\n"
else:
    output += f"- GLD (via BTC v7f): {btc_alloc*100:.0f}%\n"
for ticker, sector, mom, price in positions:
    w = stock_alloc * weight_per_stock / 100
    output += f"- {ticker}: {w*100:.1f}%\n"

output += f"""
---

## 策略 5: TQQQ v9g GLD

| 指标 | 值 |
|------|------|
| QQQ 价格 | ${qqq_current:.2f} |
| SMA200 | ${sma200:.2f} |
| QQQ vs SMA200 | {(qqq_current/sma200-1)*100:+.1f}% |
| 状态 | {tqqq_state} |
| 配置 | {tqqq_config} |

---

## 汇总

| 策略 | 当前持仓 | 关键信号 |
|------|---------|---------|
| BTC v7f DualMom | 100% {dual_mom_hold} | BTC {btc_mom*100:+.1f}% vs GLD {gld_mom*100:+.1f}% |
| Stock v4d GLD Hedge | {'70% Stock + 30% GLD' if gld_hedge_triggered else '9 只股票 (等权)'} | SPY 回撤 {spy_dd*100:.1f}% |
| Stock v3b | 9 只股票 (等权) | Top: {top3_sectors[0][0]} |
| Portfolio DD | {btc_alloc*100:.0f}% {dual_mom_hold} + {stock_alloc*100:.0f}% Stock | 回撤 {portfolio_dd*100:.1f}% |
| TQQQ v9g GLD | {tqqq_config} | QQQ {tqqq_state} |
"""

os.makedirs('portfolio/codebear', exist_ok=True)
outpath = f'portfolio/codebear/current_positions_{TODAY.replace("-","_")}.md'
with open(outpath, 'w') as f:
    f.write(output)

print(output)
print(f"\n✅ Saved to {outpath}")
