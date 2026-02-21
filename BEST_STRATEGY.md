# Best Strategies — Wombat Quant Lab

> Updated: 2026-02-21 by 代码熊 🐻

## Overall Leaderboard (by Composite Score)

> Composite = Sharpe×0.4 + Calmar×0.4 + CAGR×0.2 (simple formula)

| Rank | Strategy | CAGR | MaxDD | Sharpe | Calmar | Composite | WF |
|------|----------|------|-------|--------|--------|-----------|-----|
| 1 | **Stock v9e GDX-Compete+Vol** 🏆🆕 | **33.3%** | -14.9% | **1.64** | **2.24** | **1.617** | ✅ 0.88 |
| 2 | Stock v9d GDX-Vol ⭐⭐⭐⭐ | 32.3% | -14.9% | 1.64 | 2.17 | 1.589 | ✅ 0.88 |
| 3 | Stock v9c Vol+DD+52w+SHY ⭐⭐⭐⭐ | 31.6% | -14.9% | 1.64 | 2.12 | 1.567 | ✅ 0.89 |
| 4 | Stock v9b 52w-Hi+SHY ⭐⭐⭐ | 30.9% | -14.9% | 1.60 | 2.08 | 1.533 | ✅ 0.89 |
| 5 | Stock v9a 3m-Dom+5Sec+Breadth45 ⭐⭐⭐ | 30.5% | -14.9% | 1.57 | 2.05 | 1.512 | ✅ 0.86 |
| 6 | Stock v8d Breadth+GLD ⭐⭐ | 28.8% | -15.0% | 1.58 | 1.92 | 1.460 | ✅ 0.90 |
| 7 | Stock v4d DD+GLD ⭐⭐ | 27.1% | -15.0% | 1.45 | 1.81 | 1.356 | ✅ 0.80 |
| 8 | Stock v3b SecRot+Trend | 25.8% | -17.7% | 1.35 | 1.46 | 1.173 | ✅ 0.85 |
| 9 | Stock v3c 5Sector | 22.4% | -14.7% | 1.27 | 1.52 | 1.162 | ✅ 1.00 |
| 10 | Stock v3d Blend50 | 27.8% | -20.3% | 1.37 | 1.37 | 1.151 | ✅ 0.75 |
| 11 | BTC v7f DualMom ⭐ | 58.8% | -35.7% | 1.35 | 1.64 | 1.314 | ❌ |
| 12 | Stock v3a SecRot+Trend | 24.6% | -17.7% | 1.34 | 1.39 | 1.143 | ✅ 0.94 |

## 🏆 CURRENT CHAMPION: Stock v9e — GDX Dual-Role (Compete + Vol-Trigger)

**File**: `stocks/codebear/momentum_v9e_final.py`
**Period**: 2015-01 → 2025-12

**Key metrics**:
- CAGR **33.3%** ✅ | Sharpe **1.64** ✅ | MaxDD **-14.9%** ✅ | Calmar **2.24**
- Walk-Forward: IS 1.68, OOS 1.47, ratio **0.88** ✅
- Composite: **1.617** ✅ (vs v9c 1.567, +0.050; vs v4d 1.356, +0.261)

**Key Innovation (v9e)**: GDX serves TWO roles simultaneously:
1. **Natural Competition**: When GDX 6m-momentum ≥ 30% of avg stock → 10% GDX
   - Captures gold miner alpha during gold bull markets (48% of months)
   - GDX has 2-3x operating leverage vs GLD, amplifying gold alpha
2. **Vol-Trigger Hedge**: SPY 5-day vol > 30% → 12% GDX; > 45% → 25% GDX
   - Pre-emptive positioning before crashes fully develop

**On top of v9c foundation (8 stacked innovations)**:
- 3m-dominant momentum (50% weight), 5×2=10 stocks, breadth 45% threshold
- 52w-high proximity filter (price ≥ 60% of 52w high)  
- SHY for bear cash, GLD natural competition (70% threshold → 20%)
- Aggressive DD response (-8%→40%GLD, -12%→60%GLD, -18%→70%GLD)

**vs v9c (Previous Champion)**:
| Metric | v9c | v9e | Improvement |
|--------|-----|-----|-------------|
| CAGR | 31.6% | **33.3%** | **+1.7pp** ✅ |
| Sharpe | 1.64 | 1.64 | same |
| Calmar | 2.12 | **2.24** | **+0.12** ✅ |
| WF | 0.89 | 0.88 | -0.01 (≈same) |
| Composite | 1.567 | **1.617** | **+0.050** ✅ |

---

## Previous Champion: Stock v9c — Vol+DD+52w+SHY Complete System

**File**: `stocks/codebear/momentum_v9c_final.py`
**Period**: 2015-01 → 2025-12

**Key metrics**:
- CAGR **31.6%** ✅ | Sharpe **1.64** ✅ | MaxDD **-14.9%** ✅ | Calmar **2.12**
- Walk-Forward: IS 1.67, OOS 1.49, ratio **0.89** ✅
- Composite: **1.567** ✅ (+0.211 vs v4d original, +15.5%)

**All 9 improvements stacked in one system**:
1. [v8d] GLD natural competition (6m momentum vs stock universe)
2. [v8d] Breadth+SPY dual bear confirmation
3. [v9a] 3m-dominant momentum weights (20/50/20/10)
4. [v9a] 5 sectors × 2 stocks = 10 stocks in bull
5. [v9a] Breadth threshold 45% (slightly relaxed)
6. [v9b] **52-week high proximity filter** (price ≥ 60% of 52w high)
7. [v9b] **SHY for bear cash** (20% bear cash earns SHY returns)
8. [v9c] **SPY vol pre-emptive GLD** (5d-vol > 30%→+10%GLD, > 45%→+20%GLD)
9. [v9c] **More aggressive DD response** (-8%→40%, -12%→60%, -18%→70% GLD)

**Evolution chain**:
| Version | Composite | Δ | Key Innovation |
|---------|-----------|---|----------------|
| v4d | 1.356 | baseline | DD-responsive GLD |
| v8d | 1.460 | +0.104 | GLD competition + breadth regime |
| v9a | 1.512 | +0.052 | 3m-dominant, 5×2 sectors, breadth 45% |
| v9b | 1.533 | +0.021 | 52w-high filter + SHY cash |
| **v9c** | **1.567** | **+0.034** | **Vol pre-emptive + aggressive DD** |

**Why vol pre-emptive overlay works**:
- SPY 5-day realized vol is a cheap "VIX proxy" computable from price data
- When vol > 30% (annualized), market stress is elevated → pre-position GLD
- This fires BEFORE DD threshold is hit → reduces max intra-month drawdown
- Combined with more aggressive DD thresholds → double protection layer

---

## Stock v9b — 52w High Filter + SHY Cash

**File**: `stocks/codebear/momentum_v9b_final.py` | Composite: **1.533**, Sharpe 1.60, WF 0.89

Key additions over v9a:
- **52-week high filter**: Only select stocks trading ≥ 60% of their 52w high → filters "broken" momentum stocks
- **SHY bear cash**: 20% bear cash earns SHY returns (~4-5%/year) instead of 0%

---

## Stock v9a — First 1.5+ Breakthrough

**File**: `stocks/codebear/momentum_v9a_final.py` | Composite: **1.512**, Sharpe 1.57, WF 0.86

Key improvements over v8d:
- 3m-dominant momentum: 50% weight vs 40% previously
- 5 sectors × 2 stocks = 10 total (vs 4×3=12)
- Breadth threshold 45% (vs 40%)
- GLD competition at 70% (vs 80%)

---

## Previous Champion: Stock v9a — 3m-Dominant + 5-Sector + Breadth45 + GLD70

**🚨 FIRST TIME ALL TARGETS MET: Composite > 1.5, Sharpe > 1.5, CAGR > 30%, MaxDD < 25%, WF > 0.6 🚨**

**File**: `stocks/codebear/momentum_v9a_final.py`
**Period**: 2015-01 → 2025-12

**Key metrics**:
- CAGR **30.5%** ✅ | Sharpe **1.57** ✅ | MaxDD **-14.9%** ✅ | Calmar **2.05**
- Walk-Forward: IS 1.63, OOS 1.40, ratio **0.86** ✅
- Composite: **1.512** ✅ (vs v8d 1.460, +0.052; vs v4d 1.356, +0.156)

**Strategy**: v8d foundation + 4 synergistic improvements found by systematic sweep:

1. **3m-Dominant Momentum** (Mom_w = 1m:20%, 3m:50%, 6m:20%, 12m:10%)
   - 3-month momentum is the sweet spot for predicting next-month returns
   - Reduces 6m weight (30%→20%) to avoid over-relying on stale signals
   
2. **5-Sector Bull Mode** (5 sectors × 2 stocks = 10 stocks total)
   - vs v8d 4 sectors × 3 stocks = 12 stocks
   - Broader sector diversification captures more market breadth
   - Less per-sector concentration reduces sector-specific drawdowns
   
3. **Relaxed Breadth Threshold** (45% → was 40% in v8d)
   - More time in bull mode (lower bar for "wide breadth")
   - Avoids premature defensive positioning in moderate pullbacks
   
4. **Lower GLD Competition Bar** (70% → was 80% in v8d)
   - GLD enters naturally when momentum ≥ 70% of stock universe (vs 80%)
   - GLD participates in more defensive/uncertain periods proactively

**Key insight**: Each improvement alone adds <1% composite. Together they compound to +5.2%!

**vs v8d (Previous Champion)**:
| Metric | v8d | v9a | Improvement |
|--------|-----|-----|-------------|
| CAGR | 28.8% | **30.5%** | **+1.7pp** ✅ |
| MaxDD | -15.0% | -14.9% | +0.1pp |
| Sharpe | 1.58 | 1.57 | -0.01 (≈same) |
| Calmar | 1.92 | **2.05** | **+0.13** ✅ |
| WF | 0.90 | 0.86 | -0.04 (still great) |
| Composite | 1.460 | **1.512** | **+0.052 ✅ BREAKTHROUGH** |

---

## Previous Champion: Stock v8d — Breadth+SPY+GLD Compete+DD

**File**: `stocks/codebear/momentum_v8d_final.py`
**Period**: 2015-01 → 2025-12

**Key metrics**:
- CAGR 28.8% | Sharpe **1.58** ✅ | MaxDD -15.0% | Calmar 1.92
- Walk-Forward: IS 1.61, OOS 1.45, ratio **0.90** ✅
- Composite: **1.460** (vs v4d 1.356, +0.104)

**Strategy**: v3b sector rotation + 3-layer hedge enhancement:
1. **Dual-confirm bear regime**: Bear mode only when BOTH SPY<SMA200 AND market breadth<40%
   - More precise bear detection, avoids false triggers
2. **GLD natural competition**: GLD enters portfolio at 20% when its 6m momentum ≥ 80% of stock universe average
   - Proactive, trend-following GLD entry (not forced defensive)
3. **DD-responsive overlay**: Same v4d GLD response (-8%→30%, -12%→50%, -18%→60%)

**vs v4d (Previous Champion)**:
| Metric | v4d | v8d | Improvement |
|--------|-----|-----|-------------|
| CAGR | 27.1% | 28.8% | +1.7pp |
| MaxDD | -15.0% | -15.0% | (same) |
| Sharpe | 1.45 | **1.58** | **+0.13** ⭐ |
| Calmar | 1.81 | 1.92 | +0.11 |
| WF | 0.80 | **0.90** | **+0.10** ⭐ |
| Composite | 1.356 | **1.460** | **+0.104** ⭐ |

**Why WF improved to 0.90**:
- GLD competition mechanism is rule-based and momentum-consistent → less overfitting
- Dual-confirm regime reduces false signals that hurt OOS
- OOS Sharpe 1.45 vs IS 1.61 = only 10% degradation (excellent)

---

## Previous Champion: Stock v4d — DD Responsive GLD Hedge

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
