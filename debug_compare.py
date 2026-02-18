"""对比诊断：v3.1 策略 vs 简单全仓 DCA vs 80% 固定仓位 DCA"""
import pandas as pd
import numpy as np
from tqqq_ultimate_wombat_mode import UltimateWombat

data = UltimateWombat.generate_synthetic_data(n_years=14, seed=42)
prices = data['Adj Close']
dates = prices.index

total_deposits = 10000 + sum(1 for d in dates if d.weekday() == 4) * 1000
print(f"总投入资金: ${total_deposits:,.0f}")
print(f"标的起始价: ${prices.iloc[0]:.2f} → 终止价: ${prices.iloc[-1]:.2f}")
print(f"标的买入持有涨幅: {(prices.iloc[-1]/prices.iloc[0] - 1)*100:.1f}%\n")

# ===== v3.1 策略 =====
bot = UltimateWombat()
bot.data = data
bot.run_backtest(switch_threshold=999_999_999)  # 不触发收割，纯积累对比
v31_final = bot.results.iloc[-1]
v31_dd = ((bot.results - bot.results.cummax()) / bot.results.cummax()).min()

# ===== 简单全仓 DCA =====
cash = 10000.0; shares = 0
pv_list = []
for i in range(len(dates)):
    p = prices.iloc[i]
    if dates[i].weekday() == 4:
        cash += 1000
        shares += cash / p
        cash = 0
    pv_list.append(cash + shares * p)
simple = pd.Series(pv_list, index=dates)
simple_dd = ((simple - simple.cummax()) / simple.cummax()).min()

# ===== 80% 固定仓位 DCA (v2.0 without bear mode, like a benchmark) =====
cash = 10000.0; shares = 0; last_month = dates[0].month
pv80 = []
for i in range(len(dates)):
    p = prices.iloc[i]
    if dates[i].weekday() == 4:
        cash += 1000
    # Quarterly rebalance to 80%
    is_q = (i > 0 and dates[i].month != last_month and dates[i].month in [1,4,7,10])
    if is_q:
        tv = cash + shares * p
        target = tv * 0.80
        diff = target - shares * p
        if diff > 0:
            buy = min(diff, cash); shares += buy/p; cash -= buy
        elif diff < 0:
            sell = abs(diff); shares -= sell/p; cash += sell
    pv80.append(cash + shares * p)
    last_month = dates[i].month
fixed80 = pd.Series(pv80, index=dates)
fixed80_dd = ((fixed80 - fixed80.cummax()) / fixed80.cummax()).min()

print(f"{'策略':<25} {'最终价值':>15} {'净盈亏':>15} {'MaxDD':>10}")
print("-" * 70)
print(f"{'v3.1 连续仓位控制':<22} ${v31_final:>14,.0f} ${v31_final-total_deposits:>14,.0f} {v31_dd*100:>9.1f}%")
print(f"{'简单全仓 DCA':<22} ${simple.iloc[-1]:>14,.0f} ${simple.iloc[-1]-total_deposits:>14,.0f} {simple_dd*100:>9.1f}%")
print(f"{'80%固定仓位 DCA':<20} ${fixed80.iloc[-1]:>14,.0f} ${fixed80.iloc[-1]-total_deposits:>14,.0f} {fixed80_dd*100:>9.1f}%")
