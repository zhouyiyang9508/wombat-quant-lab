# 🐨 TQQQ Best Model — 小袋熊

## 当前最优: `beast_3q80.py` (Beast 3Q80 Mode)

### 报告数据 (TQQQ 2010-01-01 → 2026-02-18)

| 指标 | Beast 3Q80 | TQQQ Buy & Hold | SPY Benchmark |
|---|---|---|---|
| **最终价值** | **$1,106,178** (110x) | ~$850,000 (85x) | ~$55,000 (5.5x) |
| **CAGR** | **36.20%** | ~33% | ~12% |
| **Max Drawdown** | **-68.07%** | -81.66% | -33.92% |
| **Sharpe** | **0.95** | 0.85 | 1.10 |

### 策略逻辑

**牛市 (Price > SMA200):**
- 基础仓位: 80% TQQQ / 20% 现金
- 3Q 加速: 周收益 < -3% → 100% TQQQ（全仓抄底）

**熊市 (Price < SMA200):**
- 默认: 0% (100% 现金)
- Beast Bounce: RSI(10) < 30 → 80% TQQQ
- Panic Buy: RSI(10) < 20 或 周跌 > -10% → 100% TQQQ

### 其他策略

| 文件 | 描述 | 特点 |
|---|---|---|
| `beast_mode.py` | 牛市100%+熊市RSI刀法 | 更激进，全仓牛市 |
| `pro_wombat.py` | 波动率目标+SMA200 | 稳健，40%目标波动率 |

### 数据源
- yfinance (BTC-USD, TQQQ)
- 详细报告见 `tqqq_beast_3q80_report.md`

---
*Based on 小袋熊's reports, compiled by 代码熊 🐻 2026-02-19*
