#!/usr/bin/env python3
"""
动量轮动 v2c — Adaptive + Sector + Vol (Best Absolute Metrics)
代码熊 🐻

核心逻辑:
- 4因子混合动量: 0.20×1M + 0.40×3M + 0.30×6M + 0.10×12M
- 绝对动量过滤 (6M return > 0)
- 波动率过滤 (30d annualized vol < 65%)
- 行业分散: bull max 3/sector, bear max 2/sector
- Inverse-vol 加权 (风险平价思路)
- 持仓惯性 (holdover +3%)
- Dynamic sizing: bull Top 12 (100% invested), bear Top 5 (50% invested)
- Monthly rebalance

Results: CAGR 24.6%, Sharpe 1.23, MaxDD -22.0%, WF ❌ (0.65)
这是 v2d_adaptive 在 Round 1 的结果.
优势: 最高 Sharpe, 最低 MaxDD
劣势: WF 0.65 差一点 (0.70 target)
"""
# See momentum_v2_compare.py strategy_v2d for implementation
