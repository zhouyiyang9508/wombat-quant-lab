#!/usr/bin/env python3
"""
动量轮动 v2b — Low Turnover Focus
代码熊 🐻

核心逻辑:
- 双月度再平衡 (降低交易频率)
- Skip-1-month momentum (学术12-1风格, 避免短期反转)
  weights = (0, 0.40, 0.35, 0.25) → 3M/6M/12M
- 持仓惯性 (holdover bonus +5%, 减少换手)
- Regime filter (bear → cash)
- 绝对动量过滤 (6M return > 0)
- Top 10, equal weight

Results: CAGR 22.8%, Sharpe 0.92, MaxDD -25.3%, WF ✅ (1.17)
优势: WF ratio 极高, OOS > IS, 真正的样本外稳健
劣势: CAGR 和 Sharpe 低于 v1
"""
# See momentum_v2_compare.py strategy_v2b for implementation
