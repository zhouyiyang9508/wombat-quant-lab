"""
BTC v7f vs Buy & Hold 详细对比分析
代码熊 🐻
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from beast_v7f import BTCBeastV7f

BTC_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'btc_daily.csv')
GLD_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data_cache', 'GLD.csv')

# ============================================================
# Helper functions
# ============================================================
def calc_metrics(series):
    """Calculate metrics from a price/value series."""
    if len(series) < 30:
        return None
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0:
        return None
    total_ret = series.iloc[-1] / series.iloc[0] - 1
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1/years) - 1
    dd = (series - series.cummax()) / series.cummax()
    max_dd = dd.min()
    dr = series.pct_change().dropna()
    rf = 0.045 / 365
    sharpe = ((dr - rf).mean() / dr.std()) * np.sqrt(365) if dr.std() > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    vol = dr.std() * np.sqrt(365)
    return {
        'total_ret': total_ret, 'cagr': cagr, 'max_dd': max_dd,
        'sharpe': sharpe, 'calmar': calmar, 'vol': vol,
    }

def get_drawdown_recovery(series):
    """Return list of (peak_date, trough_date, recovery_date, dd_pct, recovery_days)."""
    cummax = series.cummax()
    dd = (series - cummax) / cummax
    events = []
    in_dd = False
    peak_date = trough_date = None
    trough_val = 0
    for i in range(len(dd)):
        if dd.iloc[i] < -0.10 and not in_dd:
            in_dd = True
            # find peak
            peak_idx = series.loc[:series.index[i]].idxmax()
            peak_date = peak_idx
            trough_date = series.index[i]
            trough_val = dd.iloc[i]
        elif in_dd:
            if dd.iloc[i] < trough_val:
                trough_date = series.index[i]
                trough_val = dd.iloc[i]
            if dd.iloc[i] >= 0:
                recovery_date = series.index[i]
                events.append({
                    'peak': peak_date, 'trough': trough_date,
                    'recovery': recovery_date,
                    'dd_pct': trough_val,
                    'recovery_days': (recovery_date - peak_date).days
                })
                in_dd = False
    # If still in drawdown
    if in_dd:
        events.append({
            'peak': peak_date, 'trough': trough_date,
            'recovery': None, 'dd_pct': trough_val, 'recovery_days': None
        })
    return events

# ============================================================
# Load data
# ============================================================
btc_raw = pd.read_csv(BTC_PATH, parse_dates=['Date'], index_col='Date')
btc_prices = btc_raw[['Close']].dropna().sort_index().loc['2017-01-01':'2026-02-20']['Close']

# Run v7f full period
v7f = BTCBeastV7f(initial_capital=10000)
v7f.load_data(BTC_PATH, GLD_PATH, start='2017-01-01', end='2026-02-20')
v7f.run_backtest()
v7f_series = v7f.results

# BTC B&H series (normalized to $10,000)
bh_series = btc_prices / btc_prices.iloc[0] * 10000

# Align indices
common_idx = v7f_series.index.intersection(bh_series.index)
v7f_s = v7f_series.loc[common_idx]
bh_s = bh_series.loc[common_idx]

print(f"Data range: {common_idx[0].date()} to {common_idx[-1].date()}")
print(f"v7f final: ${v7f_s.iloc[-1]:,.0f}  BH final: ${bh_s.iloc[-1]:,.0f}")

# ============================================================
# 1. OOS cumulative returns
# ============================================================
oos_periods = [
    ('2020-2026', '2020-01-01', '2026-02-20'),
    ('2021-2026', '2021-01-01', '2026-02-20'),
    ('2022-2026', '2022-01-01', '2026-02-20'),
    ('2023-2026', '2023-01-01', '2026-02-20'),
    ('2024-2026', '2024-01-01', '2026-02-20'),
]

print("\n" + "="*70)
print("1. OOS CUMULATIVE RETURNS")
print("="*70)
oos_results = []
for name, s, e in oos_periods:
    v = v7f_s.loc[s:e]
    b = bh_s.loc[s:e]
    vm = calc_metrics(v)
    bm = calc_metrics(b)
    oos_results.append((name, vm, bm))
    print(f"{name}: v7f={vm['total_ret']:+.1%}  BH={bm['total_ret']:+.1%}  v7f wins: {vm['total_ret'] > bm['total_ret']}")

# ============================================================
# 2. Sharpe comparison
# ============================================================
print("\n" + "="*70)
print("2. SHARPE RATIO COMPARISON")
print("="*70)
for name, vm, bm in oos_results:
    improve = (vm['sharpe'] - bm['sharpe']) / abs(bm['sharpe']) * 100 if bm['sharpe'] != 0 else 0
    print(f"{name}: v7f={vm['sharpe']:.2f}  BH={bm['sharpe']:.2f}  improve={improve:+.0f}%")

avg_v7f_sh = np.mean([vm['sharpe'] for _, vm, _ in oos_results])
avg_bh_sh = np.mean([bm['sharpe'] for _, _, bm in oos_results])
print(f"Average: v7f={avg_v7f_sh:.2f}  BH={avg_bh_sh:.2f}")

# ============================================================
# 3. MaxDD comparison
# ============================================================
print("\n" + "="*70)
print("3. MAX DRAWDOWN COMPARISON")
print("="*70)
for name, vm, bm in oos_results:
    improve = (abs(bm['max_dd']) - abs(vm['max_dd'])) / abs(bm['max_dd']) * 100
    print(f"{name}: v7f={vm['max_dd']:.1%}  BH={bm['max_dd']:.1%}  improve={improve:+.0f}%")

# ============================================================
# 4. $10,000 investment value at checkpoints
# ============================================================
print("\n" + "="*70)
print("4. $10,000 INVESTMENT VALUE")
print("="*70)
checkpoints = ['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01', '2025-01-01']
for cp in checkpoints:
    # Find nearest date
    v_near = v7f_s.loc[v7f_s.index >= cp].iloc[0] if len(v7f_s.loc[v7f_s.index >= cp]) > 0 else None
    b_near = bh_s.loc[bh_s.index >= cp].iloc[0] if len(bh_s.loc[bh_s.index >= cp]) > 0 else None
    if v_near is not None and b_near is not None:
        print(f"{cp}: v7f=${v_near:,.0f}  BH=${b_near:,.0f}  diff=${v_near-b_near:+,.0f}")
# Final
print(f"2026-02-20: v7f=${v7f_s.iloc[-1]:,.0f}  BH=${bh_s.iloc[-1]:,.0f}  diff=${v7f_s.iloc[-1]-bh_s.iloc[-1]:+,.0f}")

# ============================================================
# 5. Year-by-year returns
# ============================================================
print("\n" + "="*70)
print("5. YEAR-BY-YEAR RETURNS")
print("="*70)
yearly = []
for y in range(2017, 2026):
    s = f"{y}-01-01"
    e = f"{y}-12-31" if y < 2025 else "2026-02-20"
    v = v7f_s.loc[s:e]
    b = bh_s.loc[s:e]
    if len(v) < 10:
        continue
    v_ret = v.iloc[-1] / v.iloc[0] - 1
    b_ret = b.iloc[-1] / b.iloc[0] - 1
    winner = "v7f" if v_ret > b_ret else "B&H"
    yearly.append((y, v_ret, b_ret, winner))
    label = f"{y} YTD" if y == 2025 else str(y)
    print(f"{label}: v7f={v_ret:+.1%}  BH={b_ret:+.1%}  winner={winner}")

v7f_wins = sum(1 for _, _, _, w in yearly if w == "v7f")
print(f"\nv7f win rate: {v7f_wins}/{len(yearly)} = {v7f_wins/len(yearly):.0%}")

# ============================================================
# 6. Full period summary
# ============================================================
print("\n" + "="*70)
print("6. FULL PERIOD SUMMARY (2017-2026)")
print("="*70)
vm_full = calc_metrics(v7f_s)
bm_full = calc_metrics(bh_s)
for k in ['total_ret', 'cagr', 'max_dd', 'sharpe', 'calmar', 'vol']:
    print(f"{k}: v7f={vm_full[k]:.4f}  BH={bm_full[k]:.4f}")

# Best/worst years
v7f_yearly_rets = [r for _, r, _, _ in yearly]
bh_yearly_rets = [r for _, _, r, _ in yearly]
years_list = [y for y, _, _, _ in yearly]
print(f"v7f best year: {years_list[np.argmax(v7f_yearly_rets)]} ({max(v7f_yearly_rets):+.1%})")
print(f"v7f worst year: {years_list[np.argmin(v7f_yearly_rets)]} ({min(v7f_yearly_rets):+.1%})")
print(f"BH best year: {years_list[np.argmax(bh_yearly_rets)]} ({max(bh_yearly_rets):+.1%})")
print(f"BH worst year: {years_list[np.argmin(bh_yearly_rets)]} ({min(bh_yearly_rets):+.1%})")
print(f"v7f positive years: {sum(1 for r in v7f_yearly_rets if r > 0)}/{len(v7f_yearly_rets)}")
print(f"BH positive years: {sum(1 for r in bh_yearly_rets if r > 0)}/{len(bh_yearly_rets)}")

# ============================================================
# 7. Drawdown recovery
# ============================================================
print("\n" + "="*70)
print("7. DRAWDOWN RECOVERY")
print("="*70)
v7f_dd_events = get_drawdown_recovery(v7f_s)
bh_dd_events = get_drawdown_recovery(bh_s)

print("v7f drawdown events (>10%):")
for e in v7f_dd_events:
    rec = f"{e['recovery_days']} days" if e['recovery_days'] else "not recovered"
    print(f"  Peak:{e['peak'].date()} Trough:{e['trough'].date()} DD:{e['dd_pct']:.1%} Recovery:{rec}")

print("\nBH drawdown events (>10%):")
for e in bh_dd_events:
    rec = f"{e['recovery_days']} days" if e['recovery_days'] else "not recovered"
    print(f"  Peak:{e['peak'].date()} Trough:{e['trough'].date()} DD:{e['dd_pct']:.1%} Recovery:{rec}")

# ============================================================
# 8. Volatility comparison
# ============================================================
print("\n" + "="*70)
print("8. VOLATILITY COMPARISON")
print("="*70)
print(f"Full period: v7f={vm_full['vol']:.1%}  BH={bm_full['vol']:.1%}  reduction={1-vm_full['vol']/bm_full['vol']:.1%}")

# Bull periods (2017, 2019-2021, 2023-2024)
bull_dates = ['2017-01-01:2017-12-31', '2019-01-01:2021-12-31', '2023-01-01:2024-12-31']
# Bear periods (2018, 2022)
bear_dates = ['2018-01-01:2018-12-31', '2022-01-01:2022-12-31']

def period_vol(series, date_ranges):
    rets = []
    for dr in date_ranges:
        s, e = dr.split(':')
        sub = series.loc[s:e].pct_change().dropna()
        rets.extend(sub.values)
    return np.std(rets) * np.sqrt(365)

v7f_bull_vol = period_vol(v7f_s, bull_dates)
bh_bull_vol = period_vol(bh_s, bull_dates)
v7f_bear_vol = period_vol(v7f_s, bear_dates)
bh_bear_vol = period_vol(bh_s, bear_dates)
print(f"Bull periods: v7f={v7f_bull_vol:.1%}  BH={bh_bull_vol:.1%}  reduction={1-v7f_bull_vol/bh_bull_vol:.1%}")
print(f"Bear periods: v7f={v7f_bear_vol:.1%}  BH={bh_bear_vol:.1%}  reduction={1-v7f_bear_vol/bh_bear_vol:.1%}")

# ============================================================
# Generate Report
# ============================================================
report = f"""# BTC v7f vs Buy & Hold 详细对比分析报告
**代码熊 🐻 | {pd.Timestamp.now().strftime('%Y-%m-%d')}**

---

## 1. 各 OOS 期间累计收益对比

| OOS 期间 | v7f 累计收益 | BTC B&H 累计收益 | v7f 胜出 |
|----------|-------------|-----------------|---------|
"""
v7f_oos_wins = 0
for name, vm, bm in oos_results:
    win = vm['total_ret'] > bm['total_ret']
    if win: v7f_oos_wins += 1
    report += f"| {name} | {vm['total_ret']:+.1%} | {bm['total_ret']:+.1%} | {'✅' if win else '❌'} |\n"

report += f"""
**v7f 在 {v7f_oos_wins}/5 个 OOS 期间累计收益跑赢 Buy & Hold**

---

## 2. 风险调整收益对比（Sharpe Ratio）

| OOS 期间 | v7f Sharpe | BTC B&H Sharpe | 改善 |
|----------|------------|----------------|------|
"""
for name, vm, bm in oos_results:
    improve = (vm['sharpe'] - bm['sharpe']) / abs(bm['sharpe']) * 100 if bm['sharpe'] != 0 else 0
    report += f"| {name} | {vm['sharpe']:.2f} | {bm['sharpe']:.2f} | {improve:+.0f}% |\n"
report += f"| **平均** | **{avg_v7f_sh:.2f}** | **{avg_bh_sh:.2f}** | **{(avg_v7f_sh-avg_bh_sh)/abs(avg_bh_sh)*100:+.0f}%** |\n"

sharpe_wins = sum(1 for _, vm, bm in oos_results if vm['sharpe'] > bm['sharpe'])
report += f"\n**v7f 在 {sharpe_wins}/5 个 OOS 期间 Sharpe 优于 Buy & Hold**\n"

report += f"""
---

## 3. 最大回撤对比

| OOS 期间 | v7f MaxDD | BTC B&H MaxDD | 改善 |
|----------|-----------|---------------|------|
"""
for name, vm, bm in oos_results:
    improve = (abs(bm['max_dd']) - abs(vm['max_dd'])) / abs(bm['max_dd']) * 100
    report += f"| {name} | {vm['max_dd']:.1%} | {bm['max_dd']:.1%} | {improve:+.0f}% |\n"

report += f"""
---

## 4. $10,000 初始投资最终价值

| 时间点 | v7f 价值 | BTC B&H 价值 | 差异 |
|--------|---------|-------------|------|
"""
for cp in checkpoints:
    v_near = v7f_s.loc[v7f_s.index >= cp].iloc[0]
    b_near = bh_s.loc[bh_s.index >= cp].iloc[0]
    report += f"| {cp} | ${v_near:,.0f} | ${b_near:,.0f} | ${v_near-b_near:+,.0f} |\n"
report += f"| 2026-02-20 | ${v7f_s.iloc[-1]:,.0f} | ${bh_s.iloc[-1]:,.0f} | ${v7f_s.iloc[-1]-bh_s.iloc[-1]:+,.0f} |\n"

report += f"""
---

## 5. 逐年胜率统计

| 年份 | v7f 收益 | BTC B&H 收益 | 胜者 |
|------|---------|-------------|------|
"""
for y, vr, br, w in yearly:
    label = f"{y} YTD" if y == 2025 else str(y)
    report += f"| {label} | {vr:+.1%} | {br:+.1%} | {w} |\n"
report += f"\n**胜率：v7f 跑赢 Buy & Hold {v7f_wins}/{len(yearly)} 年 = {v7f_wins/len(yearly):.0%}**\n"

report += f"""
---

## 6. 关键指标汇总（全期 2017-2026）

| 指标 | v7f | BTC B&H | 胜者 |
|------|-----|---------|------|
| 累计收益 | {vm_full['total_ret']:+.1%} | {bm_full['total_ret']:+.1%} | {'v7f' if vm_full['total_ret'] > bm_full['total_ret'] else 'B&H'} |
| CAGR | {vm_full['cagr']:.1%} | {bm_full['cagr']:.1%} | {'v7f' if vm_full['cagr'] > bm_full['cagr'] else 'B&H'} |
| MaxDD | {vm_full['max_dd']:.1%} | {bm_full['max_dd']:.1%} | {'v7f' if abs(vm_full['max_dd']) < abs(bm_full['max_dd']) else 'B&H'} |
| Sharpe | {vm_full['sharpe']:.2f} | {bm_full['sharpe']:.2f} | {'v7f' if vm_full['sharpe'] > bm_full['sharpe'] else 'B&H'} |
| Calmar | {vm_full['calmar']:.2f} | {bm_full['calmar']:.2f} | {'v7f' if vm_full['calmar'] > bm_full['calmar'] else 'B&H'} |
| 年化波动率 | {vm_full['vol']:.1%} | {bm_full['vol']:.1%} | {'v7f' if vm_full['vol'] < bm_full['vol'] else 'B&H'} |
| 最差年份 | {years_list[np.argmin(v7f_yearly_rets)]}({min(v7f_yearly_rets):+.1%}) | {years_list[np.argmin(bh_yearly_rets)]}({min(bh_yearly_rets):+.1%}) | {'v7f' if min(v7f_yearly_rets) > min(bh_yearly_rets) else 'B&H'} |
| 最佳年份 | {years_list[np.argmax(v7f_yearly_rets)]}({max(v7f_yearly_rets):+.1%}) | {years_list[np.argmax(bh_yearly_rets)]}({max(bh_yearly_rets):+.1%}) | {'v7f' if max(v7f_yearly_rets) > max(bh_yearly_rets) else 'B&H'} |
| 正收益年份数 | {sum(1 for r in v7f_yearly_rets if r > 0)}/{len(v7f_yearly_rets)} | {sum(1 for r in bh_yearly_rets if r > 0)}/{len(bh_yearly_rets)} | {'v7f' if sum(1 for r in v7f_yearly_rets if r > 0) >= sum(1 for r in bh_yearly_rets if r > 0) else 'B&H'} |

---

## 7. 回撤恢复时间对比

### v7f 重大回撤事件（>10%）
| 回撤事件 | 峰值日期 | 谷底日期 | 回撤幅度 | 恢复时间 |
|----------|---------|---------|---------|---------|
"""
for e in v7f_dd_events:
    rec = f"{e['recovery_days']} 天" if e['recovery_days'] else "未恢复"
    report += f"| {e['peak'].year} | {e['peak'].date()} | {e['trough'].date()} | {e['dd_pct']:.1%} | {rec} |\n"

report += """
### BTC B&H 重大回撤事件（>10%）
| 回撤事件 | 峰值日期 | 谷底日期 | 回撤幅度 | 恢复时间 |
|----------|---------|---------|---------|---------|
"""
for e in bh_dd_events:
    rec = f"{e['recovery_days']} 天" if e['recovery_days'] else "未恢复"
    report += f"| {e['peak'].year} | {e['peak'].date()} | {e['trough'].date()} | {e['dd_pct']:.1%} | {rec} |\n"

report += f"""
---

## 8. 波动率对比

| 期间 | v7f 年化波动率 | BTC B&H 年化波动率 | 降低 |
|------|---------------|-------------------|------|
| 全期 | {vm_full['vol']:.1%} | {bm_full['vol']:.1%} | {(1-vm_full['vol']/bm_full['vol'])*100:.0f}% |
| 牛市期 | {v7f_bull_vol:.1%} | {bh_bull_vol:.1%} | {(1-v7f_bull_vol/bh_bull_vol)*100:.0f}% |
| 熊市期 | {v7f_bear_vol:.1%} | {bh_bear_vol:.1%} | {(1-v7f_bear_vol/bh_bear_vol)*100:.0f}% |

---

## 总结与建议

### 核心发现
"""

# Determine key findings
if vm_full['total_ret'] > bm_full['total_ret']:
    report += f"1. **收益**：v7f 累计收益 ({vm_full['total_ret']:+.1%}) 高于 Buy & Hold ({bm_full['total_ret']:+.1%})，CAGR 也更高\n"
else:
    report += f"1. **收益**：Buy & Hold 累计收益 ({bm_full['total_ret']:+.1%}) 高于 v7f ({vm_full['total_ret']:+.1%})\n"

report += f"2. **风险**：v7f MaxDD ({vm_full['max_dd']:.1%}) 远优于 Buy & Hold ({bm_full['max_dd']:.1%})，波动率降低 {(1-vm_full['vol']/bm_full['vol'])*100:.0f}%\n"
report += f"3. **风险调整收益**：v7f Sharpe ({vm_full['sharpe']:.2f}) vs Buy & Hold ({bm_full['sharpe']:.2f})\n"
report += f"4. **逐年胜率**：v7f 在 {v7f_wins}/{len(yearly)} 年跑赢 Buy & Hold\n"
report += f"5. **OOS 稳健性**：v7f 在 {v7f_oos_wins}/5 个 OOS 期间累计收益跑赢，{sharpe_wins}/5 个 Sharpe 跑赢\n"

report += f"""
### 给大袋熊的建议

"""
if vm_full['sharpe'] > bm_full['sharpe'] and abs(vm_full['max_dd']) < abs(bm_full['max_dd']) * 0.7:
    report += """**推荐 v7f**。理由：
- Sharpe Ratio 显著更高，说明每单位风险获得的收益更多
- MaxDD 大幅改善，熊市中回撤更小，心理压力更小
- 波动率更低，持有体验更好
- 即使收益略低于 Buy & Hold，风险调整后的表现也更优
- 对于实际投资者，能拿住的策略才是好策略——v7f 的低回撤让你不容易在底部恐慌卖出
"""
elif vm_full['total_ret'] > bm_full['total_ret']:
    report += "**推荐 v7f**。收益更高且风险更低，全面碾压 Buy & Hold。\n"
else:
    report += "**视风险偏好而定**。Buy & Hold 收益更高，但 v7f 风险调整后更优。\n"

# Write report
report_path = os.path.join(os.path.dirname(__file__), 'v7f_vs_buyhold_report.md')
with open(report_path, 'w') as f:
    f.write(report)
print(f"\n✅ Report saved to {report_path}")
