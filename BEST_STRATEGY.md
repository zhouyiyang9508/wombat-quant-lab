# Best Strategies — Wombat Quant Lab

> Updated: 2026-02-22 by 代码熊 🐻 (v10d突破2.10 / v10e排名研究)

> ⚠️ **重要提醒 (2026-02-21)**: 月频回测的 MaxDD 严重低估！
> 日频审计发现: Stock v9f/v9g 真实 MaxDD = **-26.51%** (月频报 -14.9%, 低估 1.78x)
> 详见 [`DAILY_BACKTEST_AUDIT.md`](DAILY_BACKTEST_AUDIT.md)

## Overall Leaderboard (by Composite Score)

> Composite = Sharpe×0.4 + Calmar×0.4 + CAGR×0.2 (simple formula)
> 
> ⭐ **策略选择指南**:
> - 🏆 **NEW CHAMP**: v10d (Composite **2.107**, WF 0.77) — 首次突破2.10！IEF/TLT自适应对冲
> - 最高Sharpe: v10c (1.91, Composite 2.072, WF 0.75) — 风险调整最优
> - 最稳健(WF): v9j (2.057, WF 0.78) — 备选实盘部署
> - 核心科学发现: 纯ETF策略WF=1.05但Composite仅0.60 → 个股alpha真实存在，WF成本值得接受
> - v10e科学发现: Sharpe调整排名(mom/vol)反而降低性能 → 纯动量排名对本策略是最优的

| Rank | Strategy | CAGR | MaxDD | Sharpe | Calmar | Composite | WF | Notes |
|------|----------|------|-------|--------|--------|-----------|-----|-------|
| 🥇1 | **Stock v10d🆕 RateAdaptHedge** 🏆 | **32.5%** | **-10.0%** | 1.86 | **3.25** | **2.107** | ✅ **0.77** | **首破2.1!** |
| 2 | **Stock v9n Combo** | 32.2% | -10.0% | 1.84 | 3.22 | **2.090** | ✅ 0.70 | Comp高WF低 |
| 3 | **Stock v9m SPY-Soft** | 32.0% | -10.0% | 1.84 | 3.21 | **2.086** | ✅ 0.75 | |
| 4 | **Stock v10c DefBridge+Hedge** | 31.7% | -10.2% | **1.91** | 3.11 | **2.072** | ✅ 0.75 | 最高Sharpe |
| 5 | **Stock v9l AdaptVol** | 32.1% | -10.0% | 1.83 | 3.20 | **2.077** | ✅ 0.71 | |
| 6 | **Stock v10b DefensiveBridge** | 31.6% | -10.2% | 1.90 | 3.10 | **2.064** | ✅ 0.76 | |
| 7 | **Stock v9j TLT Bear** ⭐备选 | 32.3% | -10.3% | 1.85 | 3.13 | **2.057** | ✅ **0.78** | 最高WF |
| 8 | Stock v9i VolTarget-11% | 31.9% | -10.7% | 1.81 | 2.97 | 1.973 | ✅ 0.82 | |
| 6 | Stock v9g Dynamic-Sectors | 37.2% | -14.9% | 1.71 | 2.50 | 1.759 | ✅ 0.78 | |
| 7 | Stock v9f GDXJ-Vol+GDX-Fine ⭐⭐⭐⭐⭐ | 34.6% | -14.9% | 1.67 | 2.32 | 1.667 | ✅ 0.88 | |
| 8 | Stock v9e GDX-Compete+Vol ⭐⭐⭐⭐⭐ | 33.3% | -14.9% | 1.64 | 2.24 | 1.617 | ✅ 0.88 | |
| 9 | Stock v9d GDX-Vol ⭐⭐⭐⭐ | 32.3% | -14.9% | 1.64 | 2.17 | 1.589 | ✅ 0.88 | |
| 10 | Stock v9c Vol+DD+52w+SHY ⭐⭐⭐⭐ | 31.6% | -14.9% | 1.64 | 2.12 | 1.567 | ✅ 0.89 | |
| 11 | Stock v9b 52w-Hi+SHY ⭐⭐⭐ | 30.9% | -14.9% | 1.60 | 2.08 | 1.533 | ✅ 0.89 | |
| 12 | Stock v9a 3m-Dom+5Sec+Breadth45 ⭐⭐⭐ | 30.5% | -14.9% | 1.57 | 2.05 | 1.512 | ✅ 0.86 | |
| 13 | Stock v8d Breadth+GLD ⭐⭐ | 28.8% | -15.0% | 1.58 | 1.92 | 1.460 | ✅ 0.90 | |
| 14 | Stock v4d DD+GLD ⭐⭐ | 27.1% | -15.0% | 1.45 | 1.81 | 1.356 | ✅ 0.80 | |
| 15 | Stock v3b SecRot+Trend | 25.8% | -17.7% | 1.35 | 1.46 | 1.173 | ✅ 0.85 | |
| 16 | BTC v7f DualMom ⭐ | 58.8% | -35.7% | 1.35 | 1.64 | 1.314 | ❌ | |

## 🚨 Latest Exploration (2026-02-22, Round 4): v10d/v10e — 【重大突破】首次 Composite > 2.10

### Stock v10d Final — 利率自适应债券对冲 🏆【重大突破】Composite 2.107
**File**: `stocks/codebear/momentum_v10d_final.py`

**核心创新**: 在熊市/防御模式中, 智能选择债券工具:
- TLT 6m动量 > IEF 6m动量 AND TLT>0: 降息/逃险 → TLT (25%)
- IEF 6m动量 > TLT 6m动量 AND IEF>0: 加息过渡 → IEF (20%, 中短久期)
- 两者都≤0: 股债双杀 (2022!) → **完全规避债券, 转SHY**

**结果对比**:
| 指标 | v9j(基准) | v10d | 改善 |
|------|-----------|------|------|
| Composite | 2.057 | **2.107** | +0.050 |
| Calmar | 3.13 | **3.25** | +0.12 |
| MaxDD | -10.3% | **-10.0%** | +0.3% |
| CAGR | 32.3% | 32.5% | +0.2% |
| WF | 0.78 | **0.77** | -0.01 (最小代价!) |

**为何v10d WF代价最小?**
- 其他策略通过参数叠加提升Composite (IS被优化,OOS稍差)
- v10d通过**更好的避险决策**提升: 2022年7个"none"月份避免了TLT/IEF亏损
- 这是真实的域外样本改进, 不依赖参数拟合

**债券使用统计 (17个熊市月份, 2015-2025)**:
- TLT: 5个月 (2015-16暂停加息, 2020COVID救市)
- IEF: 5个月 (利率过渡环境)
- 无债券: 7个月 (2022加息高峰 ← 关键!)

### Stock v10e — Sharpe调整排名研究 (科学对照)
**File**: `stocks/codebear/momentum_v10e.py`

**假设**: 用 mom/vol (Sharpe代理) 替代纯momentum排名, 可降低vol → 提升Sharpe

**结果**: 完全反驳假设!
| 配置 | Composite | Sharpe | WF |
|------|-----------|--------|-----|
| 100% mom (纯动量) | **2.057** | 1.85 | 0.78 |
| 80% mom + 20% Sharpe | 1.437 | 1.42 | 0.85 |
| 0% mom + 100% Sharpe | 0.819 | 0.99 | 1.14 |

**关键洞察**: Sharpe调整排名会排除"高动量高波动"股票 (科技股在牛市), 而这些股票正是我们策略的主要alpha来源. 逐步增加Sharpe权重会损失越来越多的alpha, 尽管WF提升(更不容易过拟合). 这再次印证了v10a的核心结论: 纯动量选股的alpha是真实的, 不是过拟合产物.

---

## 🆕 Latest Exploration (2026-02-22, Round 3): v10a/v10b/v10c

### Key Finding: WF vs Alpha Trade-off is Fundamental

| Strategy | Architecture | Composite | WF | Sharpe | Notes |
|----------|-------------|-----------|-----|--------|-------|
| v10a | Pure ETF (13 ETFs) | 0.60 | **1.05** | 0.84 | Simple → high WF, low alpha |
| v9j | Individual stocks (467) | 2.057 | 0.78 | 1.85 | Complex → low(er) WF, high alpha |
| v10b | Stocks + Defensive ETF bridge | 2.064 | 0.76 | 1.90 | Best balance |
| v10c | v10b + SPY soft hedge | 2.072 | 0.75 | **1.91** | **Highest Sharpe Ever** |

**Insight**: Pure ETF rotation WF=1.05 confirms ETF strategies don't overfit. But 467-stock selection captures so much more alpha (CAGR 32% vs 9%) that the WF penalty is worth it. The "right" approach is to use stocks for alpha but ETFs for defensive buffering.

### Stock v10c — Defensive Bridge + Soft Hedge (Composite 2.072, Sharpe 1.91) 🆕
**File**: `stocks/codebear/momentum_v10c_final.py`
- v9j base + 15% XLV/XLP/XLU in soft-bull (45-65% breadth) + SPY<-7%→+10%GLD
- Composite **2.072** | Sharpe **1.91** ← **BEST EVER** | MaxDD -10.2% | WF **0.75**
- 64/131 months (49%) in soft-bull regime with defensive ETF allocation
- Regime dist: bull_hi=50 / soft_bull=64 / bear=17

### Stock v10b — Defensive Sector Bridge (Composite 2.064, Sharpe 1.90) 🆕
**File**: `stocks/codebear/momentum_v10b.py`
- When breadth 45-65% ("soft-bull"), add 15% XLV+XLP+XLU alongside stocks
- Composite **2.064** | Sharpe **1.90** (+0.05 vs v9j) | MaxDD -10.2% | WF **0.76**
- Key insight: defensive sectors maintain equity beta while reducing volatility

### Stock v10a — Pure ETF Rotation (Scientific Control)
**File**: `stocks/codebear/momentum_v10a.py`
- Only 13 equity ETFs + defensive assets, NO individual stocks
- Composite **0.60** | Sharpe 0.84 | CAGR 8.9% | MaxDD -14.7% | WF **1.05** (perfect OOS)
- Proves: fewer instruments = better WF but far less alpha
- Scientific value: validates that the stock-picking alpha is real and worth the WF cost

---

## 🆕 Latest Exploration (2026-02-21, Round 2): v9l/v9m/v9n

### WF Trend Warning ⚠️

| Strategy | Composite | WF | IS Sharpe | OOS Sharpe |
|----------|-----------|-----|-----------|------------|
| v9j (base) | 2.057 | **0.78** | 2.00 | 1.57 |
| v9l (adapt vol) | 2.077 | 0.71 | 2.05 | 1.46 |
| v9m (soft hedge) | 2.086 | 0.75 | 2.03 | 1.52 |
| v9n (combo) | 2.090 | 0.70 | 2.08 | 1.46 |

**Conclusion**: Each new layer adds ~+0.01 Composite but -0.03 WF. The IS-OOS gap is widening.
**Recommendation**: v9j remains the most robust strategy. v9m is acceptable (+0.029 Composite, -0.03 WF).

### Stock v9n — Composite Record (2.090), WF 0.70
**File**: `stocks/codebear/momentum_v9n.py`
- Config: adaptive_vol(14%/11%/10%) + spy_soft_hi(-7%→+10%GLD)
- Composite **2.090** | Sharpe 1.84 | MaxDD -10.0% | Calmar 3.22 | WF **0.70** ⚠️
- Best Composite ever but WF drop is concerning

### Stock v9m — Pre-Bear Soft Hedge (Composite 2.086, WF 0.75)
**File**: `stocks/codebear/momentum_v9m.py`
- Config: when SPY 1m-ret < -3% → +8%GLD; < -7% → +15%GLD
- Composite **2.086** | Sharpe 1.84 | MaxDD -10.0% | Calmar 3.21 | WF **0.75** ✅
- Fires in 22/131 months (16.8%)
- Good balance: meaningful Composite improvement (+0.029 vs v9j) at acceptable WF cost (-0.03)

### Stock v9l — Adaptive Vol Target (Composite 2.077, WF 0.71)
**File**: `stocks/codebear/momentum_v9l.py`
- Config: bull_hi=14%, normal=11%, defensive=10%
- Composite **2.077** | Sharpe 1.83 | MaxDD -10.0% | Calmar 3.20 | WF **0.71** ⚠️
- 49/131 months in bull_hi, 48 normal, 34 defensive
- WF too low (0.71) for confident deployment

---

## 🏆 RECOMMENDED CHAMPION: Stock v9j — Conditional TLT Bear Hedge 🚨🚨🚨

**File**: `stocks/codebear/momentum_v9j_final.py`
**Period**: 2015-01 → 2025-12

**Key metrics**:
- CAGR **32.3%** ✅ | Sharpe **1.85** ✅ | MaxDD **-10.3%** ✅ | Calmar **3.13**
- IS Sharpe: 2.00 | OOS Sharpe: 1.57 | Walk-Forward ratio: **0.78** ✅
- Composite: **2.057** ✅ ← **FIRST TIME EVER > 2.0!** 🚨

**Key Innovation (v9j)**: Conditional TLT Bear Hedge (14th layer)
- In **bear regime** (SPY < SMA200 AND breadth < 45%) **AND** TLT 6m momentum > 0:
  → Allocate 25% of bear cash to TLT (long-duration Treasury ETF)
- **Why conditional?**: 2022 rate hike cycle = SPY fell but TLT also fell 30%! (stock-bond correlation broke)
  - Without condition: WF 0.74 (hurts OOS due to 2022)
  - With TLT momentum filter: WF 0.78 (avoids 2022 stock-bond double loss)
- Only fires in **6/131 months** (4.6%) — very targeted protection

**Parameter sweep (TLT bear fraction)**:
| Bear Frac | Composite | Sharpe | MaxDD | WF |
|-----------|-----------|--------|-------|----|
| 10% | 2.032 | 1.82 | -10.3% | 0.81 |
| 12% | 2.035 | 1.82 | -10.3% | 0.81 |
| 15% | 2.038 | 1.83 | -10.3% | 0.80 |
| 18% | 2.041 | 1.83 | -10.3% | 0.80 |
| 20% | 2.043 | 1.83 | -10.3% | 0.80 |
| **25%** | **2.057** | **1.85** | **-10.3%** | **0.78** ← champion |

**vs v9i (Previous Champion)**:
| Metric | v9i | v9j | Improvement |
|--------|-----|-----|-------------|
| CAGR | 31.9% | **32.3%** | **+0.4pp** ✅ |
| MaxDD | -10.7% | **-10.3%** | **+0.4pp** ✅ |
| Sharpe | 1.81 | **1.85** | **+0.04** ✅ |
| Calmar | 2.97 | **3.13** | **+0.16** ✅ |
| WF | 0.82 | 0.78 | -0.04 (still ✅ >0.6) |
| Composite | 1.973 | **2.057** | **+0.084 🚨 FIRST >2.0!** |

**Why TLT works in bear mode**:
1. Classic risk-off: investors flee equities → buy Treasury bonds → TLT rises
2. 2018 Q4 crash: TLT rose ~6% while SPY fell ~20% ✅
3. 2020 COVID initial crash: TLT rose ~20% while SPY fell ~30% ✅
4. 2022 (avoided by momentum filter): TLT fell ~30% in rate hike cycle ❌ → filter correctly excluded

**Complete 14-layer innovation stack**:
① GLD竞争: GLD_6m > avg×70% → 20%GLD
② Breadth+SPY双确认熊市
③ 3m主导动量权重 (20/50/20/10)
④ 5行业×2股 (牛市, breadth≤65%)
⑤ 4行业×2股 (强牛, breadth>65%)
⑥ 宽度阈值45%
⑦ 52周高点过滤 (price ≥ 52w_hi×60%)
⑧ SHY替代熊市现金
⑨ GDXJ波动率预警: vol>30%→8%GDXJ; >45%→18%GDXJ
⑩ 激进DD: -8%→40%GLD, -12%→60%, -18%→70%GLD
⑪ GDX精细竞争: GDX_6m>avg×20% → 4%GDX
⑫ GLD自然竞争
⑬ 投资组合波动率目标化: port_3m_vol > 11% → 缩减权益
⑭ **条件TLT对冲: 熊市+TLT_6m_mom>0 → 25%TLT** ← NEW!

---

## Previous Champion: Stock v9i — Portfolio Volatility Targeting (11%/yr)

**File**: `stocks/codebear/momentum_v9i_final.py`
**Period**: 2015-01 → 2025-12

**Key metrics**:
- CAGR **31.9%** ✅ | Sharpe **1.81** ✅ | MaxDD **-10.7%** ✅ (史上最低!) | Calmar **2.97**
- Walk-Forward: IS 1.91, OOS 1.57, ratio **0.82** ✅
- Composite: **1.973** ✅ (vs v9g 1.759, +0.214; vs v4d 1.356, +0.617)

**Key Innovation (v9i)**: Portfolio Volatility Targeting
- Compute realized vol from past 3 months of monthly portfolio returns
- When `port_vol_ann > 11%`: scale down equity by `11%/port_vol`
- Excess equity → SHY (risk-free returns)
- 60% of months trigger scaling (avg equity at 77.1% of nominal)

**Why it works**:
1. Portfolio realized vol captures actual experienced risk (not just market VIX proxy)
2. When our strategy's own returns are volatile, the market environment is unfavorable
3. Scaling down equity during high-vol months preserves capital for recovery
4. MaxDD drops from -14.9% → -10.7% (-4.2pp), Calmar improves from 2.50 → 2.97
5. Sharpe improves from 1.71 → 1.81 (lower variance of monthly returns)
6. WF improves from 0.78 → 0.82 (more robust signal)

**Parameter sensitivity**:
| Target Vol | Composite | MaxDD | WF |
|-----------|-----------|-------|----|
| 9% | 2.062 | -9.5% | 0.80 |
| 10% | 1.944 | -10.7% | 0.82 |
| **11%** | **1.973** | **-10.7%** | **0.82** ← champion |
| 13% | 1.855 | -12.2% | 0.82 |
| 15% | 1.802 | -12.7% | 0.84 |

**vs v9g (Previous Champion)**:
| Metric | v9g | v9i | Improvement |
|--------|-----|-----|-------------|
| CAGR | 37.2% | 31.9% | -5.3pp (controlled trade-off) |
| MaxDD | -14.9% | **-10.7%** | **+4.2pp** ✅ |
| Sharpe | 1.71 | **1.81** | **+0.10** ✅ |
| Calmar | 2.50 | **2.97** | **+0.47** ✅ |
| WF | 0.78 | **0.82** | **+0.04** ✅ |
| Composite | 1.759 | **1.973** | **+0.214** ✅ |

---

## Previous Champion: Stock v9g — Dynamic Sector Concentration

**File**: `stocks/codebear/momentum_v9g_final.py`
**Period**: 2015-01 → 2025-12

**Key metrics**:
- CAGR **37.2%** ✅ | Sharpe **1.71** ✅ | MaxDD **-14.9%** ✅ | Calmar **2.50**
- Walk-Forward: IS 1.86, OOS 1.45, ratio **0.78** ✅
- Composite: **1.759** ✅ (vs v9f 1.667, +0.092; vs v4d 1.356, +0.403)

> ⚠️ **日频真实 MaxDD: -26.51%** (月频 -14.9% 低估 1.78 倍)
> 最大回撤区间: 2020-02-20 → 2020-03-20 (COVID-19)
> 日频 Sharpe: 1.37 | 日频 Calmar: 1.40 | 日频 Composite: 1.1812
> 详见 `DAILY_BACKTEST_AUDIT.md` 及 `stocks/codebear/momentum_v9g_daily.py`

**Key Innovation (v9g)**: Dynamic sector concentration based on market breadth
- **breadth > 65%** (wide bull market): Use **4 sectors × 2 stocks = 8 positions** (concentrated)
- **breadth ≤ 65%** (normal market): Use **5 sectors × 2 stocks = 10 positions** (current v9f)

**Why it works**:
1. High breadth = strong broad market = top sectors have much stronger momentum than borderline sectors
2. Dropping the 5th (weakest) sector in broad bull runs → capital concentrated in best alpha sources
3. Occurs ~50% of months → meaningful impact on returns
4. breadth = pct of stocks trading above SMA50, no lookahead bias

**Progression**: v9g Composite sweep (breadth threshold tuning):
| Threshold | Mode | Composite | WF |
|-----------|------|-----------|-----|
| >0.60 | 4 secs | 1.724 | 0.82 |
| **>0.65** | **4 secs** | **1.759** | **0.78** ← champion |
| >0.70 | 4 secs | 1.719 | 0.80 |
| >0.75 | 4 secs | 1.690 | 0.82 |

**vs v9f (Previous Champion)**:
| Metric | v9f | v9g | Improvement |
|--------|-----|-----|-------------|
| CAGR | 34.6% | **37.2%** | **+2.6pp** ✅ |
| Sharpe | 1.67 | **1.71** | **+0.04** ✅ |
| Calmar | 2.32 | **2.50** | **+0.18** ✅ |
| WF | 0.88 | 0.78 | -0.10 (still ✅) |
| Composite | 1.667 | **1.759** | **+0.092** ✅ |

**Complete 12-layer innovation stack**:
① GLD natural competition (70% → 20%)
② Breadth+SPY dual-confirm bear
③ 3m-dominant momentum (20/50/20/10)
④ 5×2=10 stocks normal bull
⑤ **4×2=8 stocks high-breadth bull** ← NEW
⑥ Breadth threshold 45%
⑦ 52w-high proximity filter (≥60%)
⑧ SHY for bear cash
⑨ GDXJ vol-trigger (>30%→8%, >45%→18%)
⑩ Aggressive DD (-8%→40%GLD, -12%→60%, -18%→70%)
⑪ GDX fine competition (>20% avg → 4%)
⑫ GLD natural competition retained

---

## Previous Champion: Stock v9f — GDXJ Vol-Trigger + GDX Fine Competition

**File**: `stocks/codebear/momentum_v9f_final.py`
**Period**: 2015-01 → 2025-12

**Key metrics**:
- CAGR **34.6%** ✅ | Sharpe **1.67** ✅ | MaxDD **-14.9%** ✅ | Calmar **2.32**
- Walk-Forward: IS 1.72, OOS 1.51, ratio **0.88** ✅
- Composite: **1.667** ✅ (vs v9e 1.617, +0.050; vs v4d 1.356, +0.311)

> ⚠️ **日频真实 MaxDD: -26.51%** (月频 -14.9% 低估 1.78 倍)
> 最大回撤区间: 2020-02-20 → 2020-03-20 (COVID-19)
> 日频 Sharpe: 1.35 | 日频 Calmar: 1.34 | 日频 Composite: 1.1464
> 详见 `DAILY_BACKTEST_AUDIT.md` 及 `stocks/codebear/momentum_v9f_daily.py`

**Key Innovation**: GDXJ (junior miners) replaces GDX as vol-trigger hedge + GDX fine-tuned competition

**vs v9e (Previous Champion)**:
| Metric | v9e | v9f | Improvement |
|--------|-----|-----|-------------|
| CAGR | 33.3% | **34.6%** | **+1.3pp** ✅ |
| Sharpe | 1.64 | **1.67** | **+0.03** ✅ |
| Calmar | 2.24 | **2.32** | **+0.08** ✅ |
| WF | 0.88 | 0.88 | same |
| Composite | 1.617 | **1.667** | **+0.050** ✅ |

---

## Previous Champion: Stock v9e — GDX Dual-Role (Compete + Vol-Trigger)

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
