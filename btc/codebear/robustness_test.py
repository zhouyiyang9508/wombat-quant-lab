"""
BTC v7f vs v6b 多时间段 OOS 稳健性测试
代码熊 🐻
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from beast_v7f import BTCBeastV7f
from beast_v6b import BTCBeastV6b

BTC_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'btc_daily.csv')
GLD_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data_cache', 'GLD.csv')

def run_v7f(start, end):
    m = BTCBeastV7f()
    m.load_data(BTC_PATH, GLD_PATH, start=start, end=end)
    m.run_backtest()
    return m

def run_v6b(start, end):
    m = BTCBeastV6b()
    m.load_csv(BTC_PATH, start=start, end=end)
    m.run_backtest()
    return m

def metrics_for_period(model, start, end):
    """Get metrics for a sub-period of an already-run model."""
    r = model.results.loc[start:end]
    if len(r) < 30:
        return None
    years = (r.index[-1] - r.index[0]).days / 365.25
    if years <= 0:
        return None
    cagr = (r.iloc[-1] / r.iloc[0]) ** (1/years) - 1
    dd = (r - r.cummax()) / r.cummax()
    max_dd = dd.min()
    dr = r.pct_change().dropna()
    rf = 0.045/365
    sharpe = ((dr - rf).mean() / dr.std()) * np.sqrt(365) if dr.std() > 0 else 0
    total_ret = r.iloc[-1] / r.iloc[0] - 1
    return {'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe, 'total_ret': total_ret}

def btc_buyhold(start, end):
    btc = pd.read_csv(BTC_PATH, parse_dates=['Date'], index_col='Date')
    btc = btc[['Close']].dropna().sort_index().loc[start:end]
    if len(btc) < 30:
        return None
    r = btc['Close']
    years = (r.index[-1] - r.index[0]).days / 365.25
    if years <= 0:
        return None
    cagr = (r.iloc[-1] / r.iloc[0]) ** (1/years) - 1
    total_ret = r.iloc[-1] / r.iloc[0] - 1
    dd = (r - r.cummax()) / r.cummax()
    max_dd = dd.min()
    dr = r.pct_change().dropna()
    rf = 0.045/365
    sharpe = ((dr - rf).mean() / dr.std()) * np.sqrt(365) if dr.std() > 0 else 0
    return {'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe, 'total_ret': total_ret}

# ============================================================
# Run full backtest for both strategies over entire period
# ============================================================
print("=" * 60)
print("Running full period backtests...")
print("=" * 60)

v7f_full = run_v7f('2017-01-01', '2026-02-20')
v6b_full = run_v6b('2017-01-01', '2026-02-20')

# ============================================================
# Test 1: Multiple fixed split points
# ============================================================
print("\n" + "=" * 60)
print("TEST 1: Multiple Fixed Split Points")
print("=" * 60)

splits = [
    ('Split 1', '2017-01-01', '2019-12-31', '2020-01-01', '2026-02-20'),
    ('Split 2', '2017-01-01', '2020-12-31', '2021-01-01', '2026-02-20'),
    ('Split 3', '2017-01-01', '2021-12-31', '2022-01-01', '2026-02-20'),
    ('Split 4', '2017-01-01', '2022-12-31', '2023-01-01', '2026-02-20'),
    ('Split 5', '2017-01-01', '2023-12-31', '2024-01-01', '2026-02-20'),
]

split_results = []
for name, is_s, is_e, oos_s, oos_e in splits:
    # Run separate backtests for IS period to get proper IS metrics
    v7f_is = run_v7f(is_s, is_e)
    v6b_is = run_v6b(is_s, is_e)
    v7f_is_m = v7f_is.get_metrics()
    v6b_is_m = v6b_is.get_metrics()
    
    # Run full period and extract OOS
    v7f_oos = run_v7f(is_s, oos_e)
    v6b_oos = run_v6b(is_s, oos_e)
    v7f_oos_m = metrics_for_period(v7f_oos, oos_s, oos_e)
    v6b_oos_m = metrics_for_period(v6b_oos, oos_s, oos_e)
    
    row = {
        'name': name, 'oos_start': oos_s,
        'v7f_is_sharpe': v7f_is_m['sharpe'], 'v7f_oos_sharpe': v7f_oos_m['sharpe'],
        'v7f_wf': v7f_oos_m['sharpe'] / v7f_is_m['sharpe'] if v7f_is_m['sharpe'] != 0 else 0,
        'v7f_is_cagr': v7f_is_m['cagr'], 'v7f_is_mdd': v7f_is_m['max_dd'],
        'v7f_oos_cagr': v7f_oos_m['cagr'], 'v7f_oos_mdd': v7f_oos_m['max_dd'],
        'v6b_is_sharpe': v6b_is_m['sharpe'], 'v6b_oos_sharpe': v6b_oos_m['sharpe'],
        'v6b_wf': v6b_oos_m['sharpe'] / v6b_is_m['sharpe'] if v6b_is_m['sharpe'] != 0 else 0,
        'v6b_is_cagr': v6b_is_m['cagr'], 'v6b_is_mdd': v6b_is_m['max_dd'],
        'v6b_oos_cagr': v6b_oos_m['cagr'], 'v6b_oos_mdd': v6b_oos_m['max_dd'],
    }
    split_results.append(row)
    print(f"\n{name} (OOS: {oos_s}+)")
    print(f"  v7f: IS Sharpe={v7f_is_m['sharpe']:.2f}, OOS Sharpe={v7f_oos_m['sharpe']:.2f}, WF={row['v7f_wf']:.2f}")
    print(f"       IS CAGR={v7f_is_m['cagr']:.1%} MDD={v7f_is_m['max_dd']:.1%} | OOS CAGR={v7f_oos_m['cagr']:.1%} MDD={v7f_oos_m['max_dd']:.1%}")
    print(f"  v6b: IS Sharpe={v6b_is_m['sharpe']:.2f}, OOS Sharpe={v6b_oos_m['sharpe']:.2f}, WF={row['v6b_wf']:.2f}")
    print(f"       IS CAGR={v6b_is_m['cagr']:.1%} MDD={v6b_is_m['max_dd']:.1%} | OOS CAGR={v6b_oos_m['cagr']:.1%} MDD={v6b_oos_m['max_dd']:.1%}")

# ============================================================
# Test 2: Rolling Window
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Rolling Window (3Y IS + 1Y OOS)")
print("=" * 60)

rolling_windows = [
    ('2017-01-01', '2019-12-31', '2020-01-01', '2020-12-31'),
    ('2018-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
    ('2019-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
    ('2020-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
    ('2021-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
    ('2022-01-01', '2024-12-31', '2025-01-01', '2026-02-20'),
]

rolling_results = []
for is_s, is_e, oos_s, oos_e in rolling_windows:
    v7f_is = run_v7f(is_s, is_e)
    v6b_is = run_v6b(is_s, is_e)
    v7f_is_m = v7f_is.get_metrics()
    v6b_is_m = v6b_is.get_metrics()
    
    v7f_oos = run_v7f(is_s, oos_e)
    v6b_oos = run_v6b(is_s, oos_e)
    v7f_oos_m = metrics_for_period(v7f_oos, oos_s, oos_e)
    v6b_oos_m = metrics_for_period(v6b_oos, oos_s, oos_e)
    
    v7f_wf = v7f_oos_m['sharpe'] / v7f_is_m['sharpe'] if v7f_is_m['sharpe'] != 0 else 0
    v6b_wf = v6b_oos_m['sharpe'] / v6b_is_m['sharpe'] if v6b_is_m['sharpe'] != 0 else 0
    
    rolling_results.append({
        'is': f"{is_s[:4]}-{is_e[:4]}", 'oos': f"{oos_s[:4]}",
        'v7f_is_sh': v7f_is_m['sharpe'], 'v7f_oos_sh': v7f_oos_m['sharpe'], 'v7f_wf': v7f_wf,
        'v6b_is_sh': v6b_is_m['sharpe'], 'v6b_oos_sh': v6b_oos_m['sharpe'], 'v6b_wf': v6b_wf,
    })
    print(f"IS:{is_s[:4]}-{is_e[:4]} OOS:{oos_s[:4]} | v7f OOS Sh={v7f_oos_m['sharpe']:.2f} WF={v7f_wf:.2f} | v6b OOS Sh={v6b_oos_m['sharpe']:.2f} WF={v6b_wf:.2f}")

v7f_oos_sharpes = [r['v7f_oos_sh'] for r in rolling_results]
v6b_oos_sharpes = [r['v6b_oos_sh'] for r in rolling_results]
v7f_wfs = [r['v7f_wf'] for r in rolling_results]
v6b_wfs = [r['v6b_wf'] for r in rolling_results]

print(f"\nv7f: OOS Sharpe mean={np.mean(v7f_oos_sharpes):.2f} med={np.median(v7f_oos_sharpes):.2f} std={np.std(v7f_oos_sharpes):.2f} pass={sum(1 for w in v7f_wfs if w>=0.70)}/6")
print(f"v6b: OOS Sharpe mean={np.mean(v6b_oos_sharpes):.2f} med={np.median(v6b_oos_sharpes):.2f} std={np.std(v6b_oos_sharpes):.2f} pass={sum(1 for w in v6b_wfs if w>=0.70)}/6")

# ============================================================
# Test 3: Year-by-Year OOS
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Year-by-Year Performance")
print("=" * 60)

years = [2020, 2021, 2022, 2023, 2024, 2025]
yearly_results = []
for y in years:
    s = f"{y}-01-01"
    e = f"{y}-12-31" if y < 2025 else "2026-02-20"
    
    v7f_m = metrics_for_period(v7f_full, s, e)
    v6b_m = metrics_for_period(v6b_full, s, e)
    bh = btc_buyhold(s, e)
    
    if v7f_m and v6b_m and bh:
        v7f_win = v7f_m['total_ret'] > bh['total_ret']
        v6b_win = v6b_m['total_ret'] > bh['total_ret']
        winner = 'v7f' if v7f_m['total_ret'] > v6b_m['total_ret'] else 'v6b'
        yearly_results.append({
            'year': y,
            'v7f_ret': v7f_m['total_ret'], 'v7f_mdd': v7f_m['max_dd'], 'v7f_sharpe': v7f_m['sharpe'],
            'v6b_ret': v6b_m['total_ret'], 'v6b_mdd': v6b_m['max_dd'], 'v6b_sharpe': v6b_m['sharpe'],
            'bh_ret': bh['total_ret'], 'bh_mdd': bh['max_dd'], 'bh_sharpe': bh['sharpe'],
            'v7f_beats_bh': v7f_win, 'winner': winner,
        })
        print(f"{y}: v7f={v7f_m['total_ret']:+.1%}(MDD={v7f_m['max_dd']:.1%}) v6b={v6b_m['total_ret']:+.1%}(MDD={v6b_m['max_dd']:.1%}) BH={bh['total_ret']:+.1%} | {winner}")

v7f_beat_bh = sum(1 for r in yearly_results if r['v7f_beats_bh'])
v7f_beat_v6b = sum(1 for r in yearly_results if r['winner'] == 'v7f')
print(f"\nv7f beats B&H: {v7f_beat_bh}/{len(yearly_results)} years")
print(f"v7f beats v6b: {v7f_beat_v6b}/{len(yearly_results)} years")

# ============================================================
# Generate Report
# ============================================================
report = """# BTC v7f 多时间段 OOS 稳健性测试报告
**代码熊 🐻 | {date}**

## 背景
v7f DualMom 是当前总冠军（Composite 0.987），但只在一个 OOS（2022-2025）上验证过。
本测试在多个时间段验证其稳健性，并与 v6b（上一代冠军）对比。

---

## 测试 1：多分割点对比

| 分割点 | v7f IS Sh | v7f OOS Sh | v7f WF | v6b IS Sh | v6b OOS Sh | v6b WF |
|--------|-----------|------------|--------|-----------|------------|--------|
""".format(date=pd.Timestamp.now().strftime('%Y-%m-%d'))

for r in split_results:
    report += f"| {r['name']} ({r['oos_start'][:4]}+) | {r['v7f_is_sharpe']:.2f} | {r['v7f_oos_sharpe']:.2f} | {r['v7f_wf']:.2f} | {r['v6b_is_sharpe']:.2f} | {r['v6b_oos_sharpe']:.2f} | {r['v6b_wf']:.2f} |\n"

report += "\n### 详细 CAGR/MaxDD\n\n"
report += "| 分割点 | v7f IS CAGR | v7f IS MDD | v7f OOS CAGR | v7f OOS MDD | v6b OOS CAGR | v6b OOS MDD |\n"
report += "|--------|------------|------------|-------------|-------------|-------------|-------------|\n"
for r in split_results:
    report += f"| {r['name']} | {r['v7f_is_cagr']:.1%} | {r['v7f_is_mdd']:.1%} | {r['v7f_oos_cagr']:.1%} | {r['v7f_oos_mdd']:.1%} | {r['v6b_oos_cagr']:.1%} | {r['v6b_oos_mdd']:.1%} |\n"

report += """
---

## 测试 2：滚动窗口验证（3Y IS + 1Y OOS）

| IS 期间 | OOS 年 | v7f IS Sh | v7f OOS Sh | v7f WF | v6b IS Sh | v6b OOS Sh | v6b WF |
|---------|--------|-----------|------------|--------|-----------|------------|--------|
"""
for r in rolling_results:
    report += f"| {r['is']} | {r['oos']} | {r['v7f_is_sh']:.2f} | {r['v7f_oos_sh']:.2f} | {r['v7f_wf']:.2f} | {r['v6b_is_sh']:.2f} | {r['v6b_oos_sh']:.2f} | {r['v6b_wf']:.2f} |\n"

report += f"""
### 统计汇总

| 策略 | OOS Sharpe 均值 | 中位数 | 标准差 | WF≥0.70 通过率 |
|------|----------------|--------|--------|---------------|
| v7f | {np.mean(v7f_oos_sharpes):.2f} | {np.median(v7f_oos_sharpes):.2f} | {np.std(v7f_oos_sharpes):.2f} | {sum(1 for w in v7f_wfs if w>=0.70)}/6 |
| v6b | {np.mean(v6b_oos_sharpes):.2f} | {np.median(v6b_oos_sharpes):.2f} | {np.std(v6b_oos_sharpes):.2f} | {sum(1 for w in v6b_wfs if w>=0.70)}/6 |

---

## 测试 3：逐年表现

| 年份 | v7f 收益 | v7f MDD | v7f Sharpe | v6b 收益 | v6b MDD | v6b Sharpe | BTC B&H | 胜者(v7f vs v6b) |
|------|---------|---------|------------|---------|---------|------------|---------|----------|
"""
for r in yearly_results:
    report += f"| {r['year']} | {r['v7f_ret']:+.1%} | {r['v7f_mdd']:.1%} | {r['v7f_sharpe']:.2f} | {r['v6b_ret']:+.1%} | {r['v6b_mdd']:.1%} | {r['v6b_sharpe']:.2f} | {r['bh_ret']:+.1%} | {r['winner']} |\n"

report += f"""
- **v7f 跑赢 Buy&Hold**: {v7f_beat_bh}/{len(yearly_results)} 年
- **v7f 跑赢 v6b**: {v7f_beat_v6b}/{len(yearly_results)} 年

---

## 关键结论

### v7f OOS 表现范围
"""

oos_sharpes_v7f = [r['v7f_oos_sharpe'] for r in split_results]
oos_sharpes_v6b = [r['v6b_oos_sharpe'] for r in split_results]
report += f"- 固定分割 OOS Sharpe: {min(oos_sharpes_v7f):.2f} ~ {max(oos_sharpes_v7f):.2f}\n"
report += f"- 滚动窗口 OOS Sharpe: {min(v7f_oos_sharpes):.2f} ~ {max(v7f_oos_sharpes):.2f}\n"
report += f"- 滚动窗口平均 WF Ratio: {np.mean(v7f_wfs):.2f}\n"

worst_split = min(split_results, key=lambda x: x['v7f_oos_sharpe'])
report += f"\n### 最差 OOS\n- {worst_split['name']} (OOS from {worst_split['oos_start']}): Sharpe = {worst_split['v7f_oos_sharpe']:.2f}\n"

report += f"""
### v7f vs v6b 稳健性
- v7f 固定分割平均 OOS Sharpe: {np.mean(oos_sharpes_v7f):.2f} vs v6b: {np.mean(oos_sharpes_v6b):.2f}
- v7f 滚动窗口平均 OOS Sharpe: {np.mean(v7f_oos_sharpes):.2f} vs v6b: {np.mean(v6b_oos_sharpes):.2f}
- v7f 逐年胜率 vs v6b: {v7f_beat_v6b}/{len(yearly_results)}

### 结论
"""

# Determine conclusion
v7f_avg = np.mean(oos_sharpes_v7f)
v6b_avg = np.mean(oos_sharpes_v6b)
if v7f_avg > v6b_avg and v7f_beat_v6b > len(yearly_results) / 2:
    report += "**✅ v7f 确认为更优策略**，在多数时间段和指标上均优于 v6b，值得保持冠军地位。\n"
elif v7f_avg > v6b_avg:
    report += "**⚠️ v7f 整体略优于 v6b**，但逐年胜率不够压倒性，两者差距不大。\n"
else:
    report += "**❌ v7f 可能不如 v6b 稳健**，需要重新评估冠军地位。\n"

# Write report
report_path = os.path.join(os.path.dirname(__file__), 'v7f_robustness_report.md')
with open(report_path, 'w') as f:
    f.write(report)
print(f"\n✅ Report saved to {report_path}")
