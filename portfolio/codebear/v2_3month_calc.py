#!/usr/bin/env python3
"""Calculate 3-month performance for Portfolio v2 strategies."""
import sys, os, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from portfolio_v2_realstock import align_daily_returns
from portfolio_v2_analysis import (
    drawdown_responsive_portfolio, monthly_rp_portfolio
)

# Load returns
btc = pd.read_csv(BASE/'btc'/'codebear'/'v7f_daily_returns_2015_2025.csv', parse_dates=['Date'], index_col='Date')['Return']
stock = pd.read_csv(BASE/'stocks'/'codebear'/'v3b_daily_returns.csv', parse_dates=['Date'], index_col='Date')['Return']

print(f"BTC: {btc.index.min().date()} to {btc.index.max().date()}, n={len(btc)}")
print(f"Stock: {stock.index.min().date()} to {stock.index.max().date()}, n={len(stock)}")

# Data ends 2025-12-31, so use last ~3 months: 2025-10-01 to 2025-12-31
START = '2025-10-01'
END = '2025-12-31'
print(f"\n3-month window: {START} to {END}")

# Align
btc_a, stock_a = align_daily_returns(btc, stock)

# Build portfolio returns (full period first, then slice)
# 1. Fixed 50/50
p5050 = 0.5 * btc_a + 0.5 * stock_a
# 2. Fixed 60/40 (BTC/Stock)
p6040 = 0.6 * btc_a + 0.4 * stock_a
# 3. DD Responsive
p_dd = drawdown_responsive_portfolio(btc, stock)
# 4. Rolling Risk Parity (monthly)
p_rp = monthly_rp_portfolio(btc, stock)

def calc_metrics(ret, rf=0.04):
    """Cumulative return, max drawdown, annualized Sharpe."""
    cum = (1 + ret).prod() - 1
    # Max DD
    cumulative = (1 + ret).cumprod()
    peak = cumulative.expanding().max()
    dd = (cumulative - peak) / peak
    maxdd = dd.min()
    # Sharpe
    n_days = len(ret)
    if n_days < 5:
        return cum, maxdd, 0.0
    ann_ret = (1 + cum) ** (252 / n_days) - 1
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    return cum, maxdd, sharpe

strategies = {
    'BTC v7f': btc_a,
    'Stock v3b': stock_a,
    'DD Responsive': p_dd,
    '50/50 固定': p5050,
    '60/40 固定': p6040,
    'Rolling RP': p_rp,
}

# Full period Sharpe
full_period_sharpe = {}
for name, s in strategies.items():
    _, _, sh = calc_metrics(s.dropna())
    full_period_sharpe[name] = sh

# 3-month slice
print(f"\n{'策略':<16} {'3M收益':>8} {'3M MaxDD':>9} {'3M Sharpe':>10} {'全期Sharpe':>11} {'衰减':>8}")
print('-' * 65)

results = []
for name, s in strategies.items():
    s3 = s.loc[START:END].dropna()
    if len(s3) == 0:
        print(f"{name:<16} NO DATA")
        continue
    cum, maxdd, sharpe3 = calc_metrics(s3)
    full_sh = full_period_sharpe[name]
    decay = (sharpe3 - full_sh) / abs(full_sh) * 100 if full_sh != 0 else 0
    print(f"{name:<16} {cum:>+7.1%} {maxdd:>+8.1%} {sharpe3:>10.2f} {full_sh:>11.2f} {decay:>+7.0f}%")
    results.append((name, cum, maxdd, sharpe3, full_sh, decay))

# Write markdown
md = f"""# Portfolio v2 — 近期 3 个月表现

**时间窗口**: {START} 至 {END}（数据截止 2025-12-31）

| 策略 | 3M 收益 | 3M MaxDD | 3M Sharpe | 全期 Sharpe | Sharpe 衰减 |
|------|---------|----------|-----------|-------------|------------|
"""
for name, cum, maxdd, sh3, shf, decay in results:
    bold = "**" if name == "DD Responsive" else ""
    md += f"| {bold}{name}{bold} | {cum:+.1%} | {maxdd:+.1%} | {sh3:.2f} | {shf:.2f} | {decay:+.0f}% |\n"

md += f"""
## 分析

"""
# Find best
best = max(results, key=lambda x: x[1])
md += f"- **近期表现最好**: {best[0]}（3M 收益 {best[1]:+.1%}）\n"

# Check decay > 30%
decayed = [(n, d) for n, _, _, _, _, d in results if d < -30]
if decayed:
    md += f"- **显著衰减策略**: {', '.join(f'{n}({d:+.0f}%)' for n,d in decayed)}\n"
else:
    md += "- 无策略出现超过 30% 的 Sharpe 衰减\n"

# DD Responsive specifics
dd_row = [r for r in results if r[0] == 'DD Responsive']
if dd_row:
    dd = dd_row[0]
    md += f"- **DD Responsive**: 3M 收益 {dd[1]:+.1%}, MaxDD {dd[2]:+.1%}, 3M Sharpe {dd[3]:.2f}\n"

print("\n" + md)

out = BASE / 'portfolio' / 'codebear' / 'v2_3month_performance.md'
out.write_text(md)
print(f"\nWritten to {out}")
