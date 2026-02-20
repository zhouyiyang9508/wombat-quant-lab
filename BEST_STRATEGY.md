# Best Strategies — Wombat Quant Lab

> Updated: 2026-02-20 by 代码熊 🐻

## Overall Leaderboard (by Composite Score)

| Rank | Strategy | CAGR | MaxDD | Sharpe | Calmar | Composite |
|------|----------|------|-------|--------|--------|-----------|
| 1 | **BTC v6b 改进减半** ⭐ | 61.8% | -59.8% | 1.14 | 1.03 | 0.786 |
| 2 | BTC v5 Beast | 61.0% | -74.0% | 1.04 | 0.82 | 0.702 |
| 3 | v6a 多周期SMA | 62.9% | -70.0% | 1.09 | 0.90 | 0.742 |
| 4 | TQQQ v8 Final | 44.2% | -59.2% | 0.85 | 0.75 | 0.728 |
| 5 | Beast Rotation v1 | 38.0% | -50.7% | 0.80 | 0.75 | 0.695 |

## BTC Best: v6b 改进减半

**File**: `btc/codebear/beast_v6b.py`
**Period**: 2017-01-01 → 2026-02-18

**Key improvement**: Replace hardcoded month-based halving cycle rules with **gain-based** cycle detection:
- Gain <100% post-halving → early bull, aggressive floor (52%)
- Gain 100-300% → normal (35% floor)
- Gain >300% → late bull, conservative (70% cap in bull, 25% floor in bear)
- Gain >500% → very late (50% cap)

**Result**: MaxDD improved from -74.0% to -59.8% (14pp!), Calmar from 0.82 to 1.03.

## TQQQ Best: v8 Final

**File**: `tqqq/codebear/beast_v8.py` (or equivalent)
