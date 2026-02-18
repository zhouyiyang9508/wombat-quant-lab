"""
合成数据回测 Demo - 代码熊 🐻
用于本地验证策略指标，无需 yfinance 网络请求
"""
from tqqq_ultimate_wombat_mode import UltimateWombat
from btc_beast_3q80_mode import BTCBeastStrategy

print("=" * 55)
print("🐻 代码熊量化策略 v3.0 — 合成数据回测")
print("=" * 55)

# ===== TQQQ Strategy =====
print("\n>>> TQQQ Ultimate Wombat v3.0")
tqqq = UltimateWombat()
tqqq.data = UltimateWombat.generate_synthetic_data(n_years=14, seed=42)
tqqq.run_backtest(switch_threshold=1_000_000)
tqqq_metrics = tqqq.show_metrics()

# ===== BTC Strategy =====
print("\n>>> BTC Beast 3Q80 v3.0")
btc = BTCBeastStrategy(use_yfinance=False)
btc.df = BTCBeastStrategy.generate_synthetic_data(n_years=8, seed=99)
btc.calculate_indicators()
btc_metrics = btc.generate_report()

print("\n✅ 合成数据回测完成")
