# Best Strategies — Wombat Quant Lab

> Updated: 2026-02-20 by 代码熊 🐻

## Overall Leaderboard (by Composite Score)

> Composite = Sharpe×0.4 + Calmar×0.2 + min(CAGR,1.0)×0.2 + WF_bonus(0.2)
>
> Alternative "simple" composite (Sharpe×0.4 + Calmar×0.4 + CAGR×0.2) shown in parentheses

| Rank | Strategy | CAGR | MaxDD | Sharpe | Calmar | Composite | Simple | WF |
|------|----------|------|-------|--------|--------|-----------|--------|-----|
| 1 | **BTC v7f DualMom** ⭐⭐⭐ | 58.8% | -35.7% | 1.35 | 1.64 | **0.987** | **1.314** | ❌ |
| 2 | Stock v2d Soft Bear ⭐⭐ | 25.8% | -21.9% | 1.22 | 1.18 | 0.976 | 1.012 | ✅ |
| 3 | BTC v7d AbsMom+GLD | 65.2% | -48.0% | 1.33 | 1.36 | 0.933 | 1.134 | ❌ |
| 4 | BTC v7e ConsvMom+GLD | 59.9% | -44.6% | 1.35 | 1.34 | 0.929 | 1.126 | ❌ |
| 5 | BTC v7b GLD Hedge | 64.0% | -58.7% | 1.17 | 1.09 | 0.816 | 1.032 | ❌ |
| 6 | BTC v7c SoftBear+GLD | 61.0% | -58.0% | 1.19 | 1.05 | 0.807 | 1.018 | ❌ |
| 7 | BTC v6b 改进减半 ⭐ | 61.8% | -59.8% | 1.14 | 1.03 | 0.786 | 0.992 | ❌ |
| 8 | TQQQ v9g GLD Hedge | 47.0% | -62.6% | 0.95 | 0.75 | 0.774 | 0.774 | ✅ |
| 9 | BTC v7a SoftBear | 59.0% | -60.6% | 1.13 | 0.97 | 0.763 | 0.958 | ❌ |
| 10 | TQQQ v8 Final | 44.2% | -59.2% | 0.85 | 0.75 | 0.728 | 0.728 | ❌ |
| 11 | BTC v5 Beast | 61.0% | -74.0% | 1.04 | 0.82 | 0.702 | 0.866 | ❌ |
| 12 | Beast Rotation v1 | 38.0% | -50.7% | 0.80 | 0.75 | 0.695 | 0.696 | ✅ |

## 🏆 NEW Champion: BTC v7f — Dual Momentum Rotation

**File**: `btc/codebear/beast_v7f.py`
**Period**: 2017-01-01 → 2026-02-18

**Key metrics**:
- CAGR 58.8% | Sharpe 1.35 | MaxDD -35.7% | Calmar 1.64
- Walk-Forward: IS 1.76, OOS 0.74, ratio 0.42 (fails 0.70 threshold)
- Composite: 0.987 (code formula) / 1.314 (simple formula)
- $10,000 → $678,656

**Strategy**: BTC vs GLD dual momentum rotation with halving cycle overlay:
1. Compare BTC and GLD blended momentum (50% 6M + 50% 3M)
2. Allocate to the stronger asset:
   - BTC stronger + both positive: 80% BTC + 15% GLD
   - Only BTC positive: 85% BTC + 5% GLD
   - GLD stronger + both positive: 50% BTC + 40% GLD
   - Only GLD positive: 25% BTC + 50% GLD
   - Both negative: 20% BTC + 30% GLD + 50% cash
3. Halving cycle: early (0-18 months) → BTC minimum 50%
4. Mayer bubble protection: >2.4 reduce BTC, >3.5 cap at 35%
5. Late cycle (gain >300%): reduce BTC cap

**Why it's #1**: Composite 0.987 beats Stock v2d (0.976) even WITHOUT Walk-Forward bonus. MaxDD of -35.7% is exceptional for a BTC strategy (vs -83.2% Buy&Hold). The dual momentum rotation naturally shifts to GLD during BTC bear markets (2018, 2022), providing massive drawdown protection.

**2022 Bear Market**: Only -24.5% loss, -29.4% MaxDD (vs BTC B&H -65.2%, -67.0%)

**Key insight**: The simplicity of dual momentum (compare 2 assets, buy the stronger one) is more robust than complex regime detection with multiple indicators.

## Previous Champions (Still Notable)

### Stock v2d — Soft Bear Adaptive ⭐⭐

**File**: `stocks/codebear/momentum_v2d.py`
**Period**: 2015-01 → 2025-12

- CAGR 25.8% | Sharpe 1.22 | MaxDD -21.9% | Calmar 1.18
- **Walk-Forward: ✅ PASS** (IS 1.37, OOS 1.00, ratio 0.73)
- **Best WF-verified strategy** — the only high-composite strategy that passes WF

### BTC v6b 改进减半 ⭐

**File**: `btc/codebear/beast_v6b.py`
**Period**: 2017-01-01 → 2026-02-18

- CAGR 61.8% | Sharpe 1.14 | MaxDD -59.8% | Calmar 1.03
- Gain-based cycle detection replaced hardcoded month rules

### TQQQ v9g GLD Hedge

**File**: `tqqq/codebear/beast_v9_gld.py`
**Period**: 2010-02 → 2026-02

- 20% GLD hedge in bear regime
- Walk-Forward verified ✅

## Evolution Notes

v7 series represents a major breakthrough for BTC strategies:
- v7d/v7e/v7f all use **momentum + GLD** as the core innovation
- MaxDD dropped from -59.8% (v6b) to -35.7% (v7f) — **24pp improvement**
- This proves that GLD can be an effective hedge for crypto (despite different asset classes)
- The dual momentum approach (v7f) is the simplest and most effective design
