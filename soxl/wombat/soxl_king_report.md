# 📊 Quant Strategy Report: SOXL King (AI Wave Rider)

**Author:** Little Wombat 🐨
**Code:** `soxl_king.py`
**Date:** 2026-02-18

---

## 🦅 Executive Summary

This strategy rides the **"AI Supercycle"** using 3x Leveraged Semiconductor ETF (SOXL). It uses a classic trend-following approach (SMA 200) to capture massive bull runs (2016-2017, 2020-2021, 2023-2024) while sidestepping the brutal drawdowns of crypto-winter-like crashes (2018, 2022).

**Backtest Period:** 2012-01-01 to 2026-02-18

| Metric | SOXL King | SOXL Buy & Hold | SPY Benchmark |
| :--- | :--- | :--- | :--- |
| **Final Value** | **$421,034** (42x) | ~$250,000 (25x) | ~$55,000 (5.5x) |
| **CAGR** | **32.37%** | ~28% | ~12% |
| **Max Drawdown** | **-69.63%** | -90.45% | -33.92% |

---

## 🗡️ Core Logic

### 1. 🟢 Bull Market Engine (Price > SMA200)
- **Asset:** **100% SOXL** (3x Bull Semiconductors).
- **Rationale:** Semiconductors are the engine of modern tech growth (AI, Cloud, Crypto, EVs). When the sector runs hot, it outperforms everything. 3x leverage amplifies these runs into "life-changing wealth" events.

### 2. 🔴 Bear Market Protocol (Price < SMA200)
- **Asset:** **100% Cash**.
- **Rationale:** Semiconductor cycles are vicious. In 2022, SOXL dropped **-90%** from peak to trough. By moving to cash when the 200-day trend breaks, we preserve capital to buy back in lower.
- **Example:** Avoided the majority of the 2018 Trade War crash and the 2022 Rate Hike collapse.

---

## ⚠️ Risk Disclosure
- **Sector Concentration:** 100% exposure to a single industry (Semiconductors). Regulatory risk (China/Taiwan) or supply chain shocks can devastate the portfolio overnight.
- **Leverage Decay:** In sideways/choppy markets (whipsaws around SMA200), leverage decay will eat into returns.
- **Gap Risk:** Overnight gaps can bypass stop-loss logic.

---

## 🐨 Wombat's Verdict
Approved for "AI Thematic" allocations. Use as a satellite position alongside the core TQQQ/BTC strategies.
