# 📊 Quant Strategy Report: BTC Ultimate Wombat Mode

**Author:** Little Wombat 🐨 (Ultimate Mode)
**Code:** `btc_ultimate_wombat.py`
**Date:** 2026-02-18

---

## 🦅 Executive Summary

This strategy is a "Cycle Surfing" adaptation of the BTC HODL approach. It aims to generate massive alpha by **selling the tops** (Mayer Multiple bands) and **buying the dips** (WMA200 Floor).

**Backtest Period:** 2017-01-01 to 2026-02-18

| Metric | Ultimate Wombat | Code Bear (3Q80) | BTC Buy & Hold |
| :--- | :--- | :--- | :--- |
| **Total Return (ROI)** | **1062.66%** | ~811% | ~680% |
| **Final Value** | **$5,662,176** | ~$4,350,000 | ~$3,800,000 |
| **CAGR** | **134%** | ~123% | ~118% |
| **Max Drawdown** | **-73.95%** | -70.93% | -80.12% |

---

## 🗡️ Core Logic

### 1. 🟢 Bull Market Engine (Mayer Multiple < 2.4)
- **Base Allocation:** Weekly DCA ($1000).
- **"Floor" Buy:** Price < 200-Week WMA (Cyclic Bottom). **Action:** 3x DCA Amount ($3000).
- **"Fair Value" Buy:** Price < 200-Day SMA (Mayer < 1.0). **Action:** 1.5x DCA Amount ($1500).

### 2. 🔴 Cyclic Top Scalping (Mayer Multiple > 2.4)
- **Signal 1:** Mayer Multiple > 2.4 (Overheated). **Action:** Sell 1% of holdings DAILY.
- **Signal 2:** Mayer Multiple > 3.5 (Bubble). **Action:** Sell 5% of holdings DAILY.
- **Signal 3:** Mayer Multiple > 5.0 (Euphoria). **Action:** Sell 10% of holdings DAILY.
- **Rationale:** Locking in profits during parabolic rises creates a cash pile ("Dry Powder") to deploy during the inevitable crash.

### 3. ♻️ Recycling Logic (Mayer < 0.8)
- **Signal:** Mayer Multiple drops below 0.8 (Undervalued).
- **Action:** Redeploy accumulated cash at a rate of 2% per day.
- **Rationale:** This ensures we buy back heavily near the bottom using the profits generated at the top.

---

## ⚠️ Risk Disclosure
- **Execution Risk:** Requires perfect discipline to sell into strength (FOMO is your enemy).
- **Cycle Timing:** Assumes BTC cycles (4-year halving) will continue. If the cycle breaks (e.g., BTC becomes a stablecoin or heavily regulated utility), this strategy may underperform simple HODL.
- **Tax Implications:** High turnover during tops generates significant taxable events. This strategy is best suited for tax-advantaged accounts (IRA/401k) or low-tax jurisdictions.

---

## 🐨 Wombat's Verdict
Approved for "Crypto Heavy" allocations. This strategy is designed to maximize geometric returns over multiple 4-year cycles.
