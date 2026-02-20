# Best Strategies — Wombat Quant Lab

> Updated: 2026-02-20 by 代码熊 🐻

## Overall Leaderboard (by Composite Score)

> Composite = Sharpe×0.4 + Calmar×0.4 + CAGR×0.2

| Rank | Strategy | CAGR | MaxDD | Sharpe | Calmar | Composite | WF |
|------|----------|------|-------|--------|--------|-----------|-----|
| 1 | **Stock v2d Soft Bear** ⭐⭐ | 25.8% | -21.9% | 1.22 | 1.18 | 1.013 | ✅ |
| 2 | BTC v6b 改进减半 ⭐ | 61.8% | -59.8% | 1.14 | 1.03 | 0.992 | ❌ |
| 3 | Stock v1 Momentum | 28.9% | -24.3% | 1.10 | 1.19 | 0.976 | ❌ |
| 4 | TQQQ v9g GLD Hedge | 47.0% | -62.6% | 0.95 | 0.75 | 0.774 | ✅ |
| 5 | TQQQ v8 Final | 44.2% | -59.2% | 0.85 | 0.75 | 0.728 | ❌ |
| 6 | BTC v5 Beast | 61.0% | -74.0% | 1.04 | 0.82 | 0.702 | ❌ |
| 7 | Beast Rotation v1 | 38.0% | -50.7% | 0.80 | 0.75 | 0.695 | ✅ |

## 🏆 NEW Champion: Stock v2d — Soft Bear Adaptive

**File**: `stocks/codebear/momentum_v2d.py`
**Period**: 2015-01 → 2025-12 (S&P 500 universe)

**Key metrics**:
- CAGR 25.8% | Sharpe 1.22 | MaxDD -21.9% | Calmar 1.18
- Walk-Forward: IS 1.37, OOS 1.00, **ratio 0.73 ✅**
- Turnover: 48.5%/month

**Strategy**: Monthly momentum rotation of S&P 500 stocks with:
1. 4-factor blended momentum (1M/3M/6M/12M)
2. Absolute momentum filter (6M > 0)
3. Volatility filter (30d vol < 65%)
4. Sector diversification (max 3/sector bull, 2/sector bear)
5. Inverse-vol weighting
6. Holdover bonus (+3%)
7. **Soft bear regime**: Bull=Top12 100%, **Bear=Top8 80%** invested

**Why it's #1**: Highest Composite (1.013) AND passes Walk-Forward. The "soft bear" design avoids the trap of over-hedging during market downturns while still providing meaningful drawdown protection.

## BTC Best: v6b 改进减半

**File**: `btc/codebear/beast_v6b.py`
**Period**: 2017-01-01 → 2026-02-18

**Key improvement**: Replace hardcoded month-based halving cycle rules with **gain-based** cycle detection:
- Gain <100% post-halving → early bull, aggressive floor (52%)
- Gain 100-300% → normal (35% floor)
- Gain >300% → late bull, conservative (70% cap in bull, 25% floor in bear)
- Gain >500% → very late (50% cap)

**Result**: MaxDD improved from -74.0% to -59.8% (14pp!), Calmar from 0.82 to 1.03.

## TQQQ Best: v9g GLD Hedge

**File**: `tqqq/codebear/beast_v9_gld.py`
**Period**: 2010-02 → 2026-02

**Key improvement**: Add 20% GLD hedge allocation in bear regime, reducing MaxDD while maintaining CAGR.
Walk-Forward verified ✅.
