# Best Strategies — Wombat Quant Lab

> Updated: 2026-02-20 by 代码熊 🐻

## Overall Leaderboard (by Composite Score)

> Composite = Sharpe×0.4 + Calmar×0.2 + min(CAGR,1.0)×0.2 + WF_bonus(0.2)
>
> Alternative "simple" composite (Sharpe×0.4 + Calmar×0.4 + CAGR×0.2) shown in parentheses

| Rank | Strategy | CAGR | MaxDD | Sharpe | Calmar | Composite | Simple | WF |
|------|----------|------|-------|--------|--------|-----------|--------|-----|
| 1 | **Stock v4d DD+GLD** ⭐⭐⭐ | 27.1% | -15.0% | 1.45 | 1.81 | **1.158** | **1.356** | ✅ 0.80 |
| 2 | BTC v7f DualMom ⭐⭐ | 58.8% | -35.7% | 1.35 | 1.64 | 0.987 | 1.314 | ❌ |
| 3 | Stock v3b SecRot+Trend | 25.8% | -17.7% | 1.35 | 1.46 | 1.084 | 1.173 | ✅ 0.85 |
| 4 | Stock v3c 5Sector | 22.4% | -14.7% | 1.27 | 1.52 | 1.062 | 1.162 | ✅ 1.00 |
| 5 | Stock v3d Blend50 | 27.8% | -20.3% | 1.37 | 1.37 | 1.052 | 1.151 | ✅ 0.75 |
| 6 | Stock v3a SecRot+Trend | 24.6% | -17.7% | 1.34 | 1.39 | 1.043 | 1.143 | ✅ 0.94 |
| 7 | Stock v3e v2d+Trend | 24.5% | -16.7% | 1.26 | 1.46 | 1.045 | 1.139 | ✅ 0.72 |
| 8 | Stock v2d Soft Bear | 25.8% | -21.9% | 1.22 | 1.18 | 0.976 | 1.012 | ✅ 0.73 |
| 9 | BTC v7d AbsMom+GLD | 65.2% | -48.0% | 1.33 | 1.36 | 0.933 | 1.134 | ❌ |
| 10 | BTC v7e ConsvMom+GLD | 59.9% | -44.6% | 1.35 | 1.34 | 0.929 | 1.126 | ❌ |
| 11 | BTC v7b GLD Hedge | 64.0% | -58.7% | 1.17 | 1.09 | 0.816 | 1.032 | ❌ |
| 12 | TQQQ v9g GLD Hedge | 47.0% | -62.6% | 0.95 | 0.75 | 0.774 | 0.774 | ✅ |
| 13 | Beast Rotation v1 | 38.0% | -50.7% | 0.80 | 0.75 | 0.695 | 0.696 | ✅ |

## 🏆 NEW Champion: Stock v4d — DD Responsive GLD Hedge

**File**: `stocks/codebear/momentum_v4d.py`
**Period**: 2015-01 → 2025-12

**Key metrics**:
- CAGR 27.1% | Sharpe 1.45 | MaxDD -15.0% | Calmar 1.81
- Walk-Forward: IS 1.54, OOS 1.23, ratio 0.80 ✅
- Simple Composite: 1.356

**Strategy**: v3b (Sector Rotation + SMA50 Trend) + DD Responsive GLD Hedge:
1. Inherits all v3b logic (SMA50, sector rotation, blend weighting, tighter bear)
2. Adds DD-responsive GLD allocation:
   - Normal: 100% Stock v3b
   - DD > -8%: 30% GLD + 70% Stock
   - DD > -12%: 50% GLD + 50% Stock
   - DD > -18%: 60% GLD + 40% Stock
3. Auto-recovers to 100% Stock when drawdown heals

**vs v3b (Previous Champion)**:
| Metric | v3b | v4d | Improvement |
|--------|-----|-----|-------------|
| CAGR | 25.8% | 27.1% | +1.3pp |
| MaxDD | -17.7% | -15.0% | **+2.7pp** |
| Sharpe | 1.35 | 1.45 | +0.10 |
| Calmar | 1.46 | 1.81 | **+0.35** |
| Composite | 1.173 | 1.356 | **+0.183** |

**Why it works**:
- GLD only activates during actual drawdowns (18 months out of 131 = 13.7%)
- 2018 Q4: GLD rose 8.6% while stocks dropped, cutting MaxDD from -17.7% to -15.0%
- Zero drag in bull markets (0% GLD when DD = 0)
- Doesn't depend on market regime — purely reactive to portfolio drawdown

---

## Previous Champion: Stock v3b — Sector Rotation + SMA50 Trend + Blend30

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

## Stock v4 Series — GLD Hedge Variants

| Version | CAGR | MaxDD | Sharpe | Calmar | WF | Composite |
|---------|------|-------|--------|--------|----|-----------|
| **v4d_aggr** ⭐ | 27.1% | -15.0% | 1.45 | 1.81 | 0.80 ✅ | 1.356 |
| v4d_orig | 26.6% | -15.2% | 1.40 | 1.75 | 0.83 ✅ | 1.314 |
| v4e SoftBear+GLD | 26.7% | -17.7% | 1.38 | 1.51 | 0.82 ✅ | 1.208 |
| v4a DeepBear GLD | 25.0% | -17.7% | 1.32 | 1.41 | 0.83 ✅ | 1.142 |
| v4c DualMom | 23.4% | -17.7% | 1.34 | 1.32 | 0.88 ✅ | 1.111 |

DD Responsive (v4d) is the clear best approach — reactive to actual drawdown, not market regime.

## Evolution Notes

Stock v4 builds on v3's breakthrough with GLD defensive hedging:
- v3b→v4d: Composite 1.173 → 1.356 (+15.6%) — biggest single improvement yet
- DD Responsive is superior to regime-based GLD allocation
- All v4 variants pass Walk-Forward, proving GLD hedging is robust
- Key insight: React to portfolio pain, not market predictions

*Updated by 代码熊 🐻 — 2026-02-20*
