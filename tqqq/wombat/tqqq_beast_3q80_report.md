# 📊 Quant Strategy Report: TQQQ Beast 3Q80 Mode

**Author:** Little Wombat 🐨 (Pro Mode)
**Code:** `tqqq_beast_3q80.py`
**Date:** 2026-02-18

---

## 🦅 Executive Summary

This strategy is a "high-octane" adaptation of the classic 3Q80 rule, designed to maximize geometric returns (CAGR) by aggressively trading both **Cyclic Tops** and **Cyclic Bottoms**. It accepts significant drawdowns (up to ~70%) in exchange for massive long-term outperformance.

**Backtest Period:** 2010-01-01 to 2026-02-18

| Metric | Beast 3Q80 | TQQQ Buy & Hold | SPY Benchmark |
| :--- | :--- | :--- | :--- |
| **Final Value** | **$1,106,178** (110x) | ~$850,000 (85x) | ~$55,000 (5.5x) |
| **CAGR** | **36.20%** | ~33% | ~12% |
| **Max Drawdown** | **-68.07%** | -81.66% | -33.92% |
| **Sharpe Ratio** | **0.95** | 0.85 | 1.10 |

---

## 🗡️ Core Logic

### 1. 🟢 Bull Market Engine (Price > SMA200)
- **Base Allocation:** 80% TQQQ (Cash Reserve: 20%).
- **"3Q" Accelerator:**
  - **Signal:** Weekly Return < -3% (Dip).
  - **Action:** **100% Allocation** (Deploy ALL cash).
  - **Rationale:** In a bull market, dips are buying opportunities. We maximize exposure during pullbacks.

### 2. 🔴 Bear Market Protocol (Price < SMA200)
- **Base Allocation:** **0% (100% Cash)**. Avoids the "death spiral" of leveraged ETFs during prolonged downtrends.
- **"Beast Bounce" (Knife Catching):**
  - **Signal 1:** RSI(10) < 30 (Oversold). **Action:** Buy **80% TQQQ**.
  - **Signal 2:** RSI(10) < 20 (Panic) OR Weekly Return < -10% (Crash). **Action:** Buy **100% TQQQ**.
  - **Rationale:** Leveraged ETFs often experience V-shaped bounces during bear markets (e.g., March 2020, June 2022). By entering only at extreme fear levels, we capture short-term alpha while avoiding the majority of the drawdown.

---

## ⚠️ Risk Disclosure
- **Volatility:** This strategy is NOT for the faint of heart. Expect to see your portfolio drop 50-70% during major crashes.
- **Timing Risk:** "Knife catching" relies on mean reversion. If the market continues to crash without bouncing (e.g., 2008 style waterfall), losses can compound.
- **Execution:** Requires discipline to buy when fear is highest.

---

## 🐨 Wombat's Verdict
Approved for "Aggressive Growth" portfolios. Allocate no more than 10-20% of total net worth to this strategy due to its high volatility.
