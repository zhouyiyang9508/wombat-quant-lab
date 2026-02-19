"""多 seed 对比测试：v4.0 vs 80%固定 DCA vs 简单全仓 DCA"""
import pandas as pd
import numpy as np
from tqqq_ultimate_wombat_mode import UltimateWombat

seeds = [42, 123, 777, 2024, 3141, 9999, 54321, 88888]

print(f"{'Seed':>7} | {'v4.0 净盈亏':>14} {'MaxDD':>8} | {'80%固定':>14} {'MaxDD':>8} | {'全仓DCA':>14} {'MaxDD':>8} | {'v4胜?':>5}")
print("-" * 100)

v4_wins = 0
for seed in seeds:
    data = UltimateWombat.generate_synthetic_data(n_years=14, seed=seed)
    prices = data['Adj Close']
    dates = prices.index
    total_dep = 10000 + sum(1 for d in dates if d.weekday() == 4) * 1000

    # v4.0
    bot = UltimateWombat()
    bot.data = data
    bot.run_backtest(switch_threshold=999_999_999)
    v4_final = bot.results.iloc[-1]
    v4_dd = ((bot.results - bot.results.cummax()) / bot.results.cummax()).min()

    # 80% Fixed
    cash = 10000.0; shares = 0; lm = dates[0].month; pv80 = []
    for i in range(len(dates)):
        p = prices.iloc[i]
        if dates[i].weekday() == 4: cash += 1000
        is_q = (i > 0 and dates[i].month != lm and dates[i].month in [1,4,7,10])
        if is_q:
            tv = cash + shares * p
            tgt = tv * 0.80; diff = tgt - shares * p
            if diff > 0: buy = min(diff, cash); shares += buy/p; cash -= buy
            elif diff < 0: sell = abs(diff); shares -= sell/p; cash += sell
        pv80.append(cash + shares * p)
        lm = dates[i].month
    f80 = pd.Series(pv80, index=dates)
    f80_final = f80.iloc[-1]
    f80_dd = ((f80 - f80.cummax()) / f80.cummax()).min()

    # Simple DCA
    cash = 10000.0; shares = 0; pvs = []
    for i in range(len(dates)):
        p = prices.iloc[i]
        if dates[i].weekday() == 4: cash += 1000; shares += cash/p; cash = 0
        pvs.append(cash + shares * p)
    sdca = pd.Series(pvs, index=dates)
    sdca_final = sdca.iloc[-1]
    sdca_dd = ((sdca - sdca.cummax()) / sdca.cummax()).min()

    win = "✅" if v4_final > f80_final else "❌"
    if v4_final > f80_final: v4_wins += 1
    print(f"{seed:>7} | ${v4_final-total_dep:>12,.0f} {v4_dd*100:>7.1f}% | ${f80_final-total_dep:>12,.0f} {f80_dd*100:>7.1f}% | ${sdca_final-total_dep:>12,.0f} {sdca_dd*100:>7.1f}% | {win}")

print(f"\nv4.0 胜率: {v4_wins}/{len(seeds)} ({v4_wins/len(seeds)*100:.0f}%)")
