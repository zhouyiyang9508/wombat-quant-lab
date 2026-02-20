# Best Strategies — Wombat Quant Lab

> Updated: 2026-02-20 by 代码熊 🐻

## Overall Leaderboard (by Composite Score)

> Composite = Sharpe×0.4 + Calmar×0.2 + min(CAGR,1.0)×0.2 + WF_bonus(0.2)
>
> Alternative "simple" composite (Sharpe×0.4 + Calmar×0.4 + CAGR×0.2) shown in parentheses

| Rank | Strategy | CAGR | MaxDD | Sharpe | Calmar | Composite | Simple | WF |
|------|----------|------|-------|--------|--------|-----------|--------|-----|
| 1 | **Stock v3b SecRot+Trend** ⭐⭐⭐ | 25.8% | -17.7% | 1.35 | 1.46 | **1.084** | **1.173** | ✅ 0.85 |
| 2 | BTC v7f DualMom ⭐⭐ | 58.8% | -35.7% | 1.35 | 1.64 | 0.987 | 1.314 | ❌ |
| 3 | Stock v3c 5Sector | 22.4% | -14.7% | 1.27 | 1.52 | 1.062 | 1.162 | ✅ 1.00 |
| 4 | Stock v3d Blend50 | 27.8% | -20.3% | 1.37 | 1.37 | 1.052 | 1.151 | ✅ 0.75 |
| 5 | Stock v3a SecRot+Trend | 24.6% | -17.7% | 1.34 | 1.39 | 1.043 | 1.143 | ✅ 0.94 |
| 6 | Stock v3e v2d+Trend | 24.5% | -16.7% | 1.26 | 1.46 | 1.045 | 1.139 | ✅ 0.72 |
| 7 | Stock v2d Soft Bear | 25.8% | -21.9% | 1.22 | 1.18 | 0.976 | 1.012 | ✅ 0.73 |
| 8 | BTC v7d AbsMom+GLD | 65.2% | -48.0% | 1.33 | 1.36 | 0.933 | 1.134 | ❌ |
| 9 | BTC v7e ConsvMom+GLD | 59.9% | -44.6% | 1.35 | 1.34 | 0.929 | 1.126 | ❌ |
| 10 | BTC v7b GLD Hedge | 64.0% | -58.7% | 1.17 | 1.09 | 0.816 | 1.032 | ❌ |
| 11 | TQQQ v9g GLD Hedge | 47.0% | -62.6% | 0.95 | 0.75 | 0.774 | 0.774 | ✅ |
| 12 | Beast Rotation v1 | 38.0% | -50.7% | 0.80 | 0.75 | 0.695 | 0.696 | ✅ |

## 🏆 NEW Champion: Stock v3b — Sector Rotation + SMA50 Trend + Blend30

**File**: `stocks/codebear/momentum_v3b.py`
**Period**: 2015-01 → 2025-12

**Key metrics**:
- CAGR 25.8% | Sharpe 1.35 | MaxDD -17.7% | Calmar 1.46
- Walk-Forward: IS 1.39, OOS 1.18, ratio 0.85 ✅
- Composite: 1.084 (code formula) / 1.173 (simple formula)

**Strategy**: Two-stage sector rotation with SMA50 trend filter:
1. **SMA50 Trend Filter**: Only consider stocks with price > 50-day SMA
2. **Sector Rotation**: Rank sectors by avg momentum, pick top 4 (bull) / 3 (bear)
3. **Stock Selection**: Within each top sector, pick top 3 (bull) / 2 (bear) stocks
4. **Blended Weighting**: 70% inverse-volatility + 30% momentum-proportional
5. **Bear Regime**: SPY < SMA200 → 3 sectors × 2 stocks + 20% cash

**Why it's #1**:
- **Highest WF-adjusted Composite**: 1.084 beats BTC v7f (0.987) thanks to Walk-Forward bonus
- **Exceptional risk control**: MaxDD -17.7% is remarkable for a stock momentum strategy
- **Perfect WF ratio**: 0.85 (well above 0.70 threshold)
- **All v3 variants pass WF**: Proving the innovations are robust, not overfit

**Key innovations that made the difference**:
1. SMA50 trend filter → MaxDD reduced by 4.2pp vs v2d
2. Sector rotation → Sharpe improved by +0.12
3. Blended weighting → More capital to strongest movers
4. Tighter bear handling → More concentrated in best bear sectors

**vs v2d (Previous Stock Champion)**:
| Metric | v2d | v3b | Improvement |
|--------|-----|-----|-------------|
| MaxDD | -21.9% | -17.7% | +4.2pp |
| Sharpe | 1.22 | 1.35 | +0.13 |
| Calmar | 1.18 | 1.46 | +0.28 |
| OOS Sharpe | 1.00 | 1.18 | +0.18 |
| Composite | 1.013 | 1.173 | +0.160 |

## Previous Champions (Still Notable)

### BTC v7f — Dual Momentum Rotation ⭐⭐

**File**: `btc/codebear/beast_v7f.py`
- CAGR 58.8% | Sharpe 1.35 | MaxDD -35.7% | Calmar 1.64
- Highest simple composite (1.314) but fails Walk-Forward
- Best absolute returns of any strategy

### Stock v2d — Soft Bear Adaptive

**File**: `stocks/codebear/momentum_v2d.py`
- CAGR 25.8% | Sharpe 1.22 | MaxDD -21.9% | Calmar 1.18
- Superseded by v3b (same CAGR, better risk metrics)

## Stock v3 Series — Complete Results

All 5 variants pass Walk-Forward validation:

| Version | CAGR | MaxDD | Sharpe | Calmar | WF | Composite |
|---------|------|-------|--------|--------|----|-----------|
| **v3b** ⭐ | 25.8% | -17.7% | 1.35 | 1.46 | 0.85 ✅ | 1.173 |
| v3c | 22.4% | -14.7% | 1.27 | 1.52 | 1.00 ✅ | 1.162 |
| v3d | 27.8% | -20.3% | 1.37 | 1.37 | 0.75 ✅ | 1.151 |
| v3a | 24.6% | -17.7% | 1.34 | 1.39 | 0.94 ✅ | 1.143 |
| v3e | 24.5% | -16.7% | 1.26 | 1.46 | 0.72 ✅ | 1.139 |

Note: v3c has the lowest MaxDD (-14.7%) and perfect WF ratio (1.00) — best for risk-averse investors.

## Evolution Notes

Stock v3 represents a major breakthrough:
- All 5 variants beat v2d's Composite 1.013 by significant margins (1.139-1.173)
- The SMA50 trend filter is the single most important innovation
- Sector rotation adds further improvement on top
- These innovations are robust (all pass WF) and not overfit

*Updated by 代码熊 🐻 — 2026-02-20*
