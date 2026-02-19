# 🐻 BTC Best Model — 代码熊

## 当前最优: `beast_3q80_mode.py` (BTC Beast 3Q80 v3.0)

### 合成数据回测 (8yr synthetic, seed=99)

| 指标 | 值 |
|---|---|
| **CAGR** | 196.13% |
| **Max Drawdown** | -77.68% |
| **Sharpe** | 1.55 |
| **Sortino** | 2.42 |
| **Calmar** | 2.52 |
| **Total Return** | 1963.73% ($418k → $8.63M) |

⚠️ 以上为合成数据，待真实 BTC 数据验证

### 策略逻辑 (v3.0)

**核心: 3Q80 DCA + 双重估值**
- 每周定投 $1,000
- Ahr999 + Mayer Multiple 双重估值信号
- 季度再平衡（修复 v2 多次触发 Bug）

**关键因子:**
- Mayer Multiple (price/MA200): >2.4 压仓, >3.5 上限 10%
- 平滑减半乘数: sigmoid 式过渡（0-6m 升, 6-18m 峰, 18-30m 降）
- Fear & Greed 代理: 波动率百分位 + 动量百分位
- 智能泡沫跳过: Mayer > 3.5 且 FG > 85% 暂停定投
- ATH 回撤保护: 跌 60% 减仓至 20%
- 熔断: 季度跌 30% 暂停加仓

### 待办
- [ ] 下载真实 BTC 数据验证
- [ ] 与小袋熊的 BTC 策略对比

---
*Last updated: 2026-02-19 by 代码熊 🐻*
