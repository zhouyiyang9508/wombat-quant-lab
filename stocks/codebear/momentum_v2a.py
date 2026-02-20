#!/usr/bin/env python3
"""
动量轮动 v2a — Regime Filter + 绝对动量
代码熊 🐻

核心逻辑:
- Bull (SPY > SMA200): Top 10 momentum, 6M abs return > 0, equal weight
- Bear (SPY < SMA200): 100% cash
- Monthly rebalance

Results: CAGR 21.3%, Sharpe 0.97, MaxDD -25.1%, WF ❌ (0.42)
失败原因: Bear→全现金太激进, 错过反弹
"""
# See momentum_v2_compare.py strategy_v2a for implementation
# This file documents the strategy logic and results
