#!/usr/bin/env python3
"""
Hybrid v5 — 动态 BTC/ETH 权重优化
代码熊 🐻 | 2026-02-22

基于 v3 架构，专注改进 Crypto 组件内部的 BTC/ETH 权重分配：

v3 问题：
  - BTC/ETH 轮换是"赢家通吃"型（动量高者拿70%）
  - 200dMA 过滤由 BTC 主导：BTC<MA → 整个 crypto → GLD
    → ETH 可能还在趋势中，但跟着 BTC 一起退出，浪费了 ETH 的收益

v5 改进：
  策略 A: 动量比例权重（连续，非离散）
    btc_w = softmax([btc_mom, eth_mom])[0]，clip到[0.25,0.75]
    
  策略 B: 各自独立 200dMA 过滤
    - BTC < BTC_200dMA → BTC allocation → GLD
    - ETH < ETH_200dMA → ETH allocation → GLD
    - 两个独立过滤，互不影响
    
  策略 C: A + B（连续权重 + 独立过滤）← 核心创新

  策略 D: 动量阈值切换（动量差异>阈值才切换，否则保持50:50）
    避免频繁换手

扫描：
  - 策略类型 × w_crypto [0.10~0.35]
  - 目标: WF≥0.58, CAGR≥50%, MaxDD<30%

股票组件: 直接复用 v3 的 StockV9j equity 曲线（已验证）
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parents[2]
CACHE = BASE / "data_cache"

# Crypto 参数（继承 v3）
CRYPTO_MA_WIN     = 200
CRYPTO_DD_EXIT    = -0.25
CRYPTO_DD_REENTRY = -0.15
CRYPTO_VOL_WIN    = 60
CRYPTO_VOL_TGT    = 0.80
CRYPTO_VOL_MIN    = 0.40
CRYPTO_VOL_MAX    = 1.50
CRYPTO_MOM_LB     = 90   # 动量窗口（天）
MOM_SWITCH_THRESH = 0.05  # 策略D: 动量差异<5% → 保持50:50
COST = 0.0015

def load_series(name):
    fp = CACHE / f"{name}.csv"
    df = pd.read_csv(fp)
    c = 'Date' if 'Date' in df.columns else df.columns[0]
    df[c] = pd.to_datetime(df[c])
    df = df.set_index(c).sort_index()
    col = 'Close' if 'Close' in df.columns else df.columns[0]
    return pd.to_numeric(df[col], errors='coerce').dropna()

def softmax_split(score_a, score_b, lo=0.25, hi=0.75):
    """连续动量权重，clip 到 [lo, hi] 避免过度集中"""
    if np.isnan(score_a) or np.isnan(score_b):
        return 0.5, 0.5
    # 平移到正值后做比例
    mn = min(score_a, score_b)
    a = score_a - mn + 1e-8
    b = score_b - mn + 1e-8
    wa = float(np.clip(a / (a + b), lo, hi))
    return wa, 1.0 - wa

def run_crypto_v5_daily(btc_p, eth_p, gld_p,
                         strategy='C',       # A/B/C/D
                         mom_w=(0.20,0.50,0.20,0.10),
                         start='2015-01-01', end='2025-12-31'):
    """
    Crypto v5 — 动态 BTC/ETH 权重
    strategy:
      'A' = 连续动量比例权重（固定 BTC+ETH 各有一个MA，只要一个低于MA就不再持有该币）
            → 但两个独立，BTC跌破不影响ETH
      'B' = 各自独立MA过滤（50:50默认，仅跌破才退出该部分）
      'C' = A + B（连续权重 + 独立MA过滤）← 主打
      'D' = 动量阈值：差异>THRESH用80:20，否则50:50（+ 独立MA过滤）
      'v3'= 复现v3逻辑（验证用）
    """
    # 公共交易日
    idx = btc_p.loc[start:end].dropna().index.union(
          eth_p.loc[start:end].dropna().index)
    trading_days = idx

    month_ends = pd.Series(1, index=trading_days).resample('ME').last().index

    # 预计算
    btc_ma  = btc_p.rolling(CRYPTO_MA_WIN).mean()
    eth_ma  = eth_p.rolling(CRYPTO_MA_WIN).mean()
    btc_vol = np.log(btc_p/btc_p.shift(1)).rolling(CRYPTO_VOL_WIN).std()*np.sqrt(252)

    def mom_score(prices, date, lb):
        h = prices.loc[:date].dropna()
        if len(h) < lb + 5: return np.nan
        return float(h.iloc[-1] / h.iloc[-lb] - 1)

    def multi_mom(prices, date):
        """同股票策略的多周期动量加权"""
        h = prices.loc[:date].dropna()
        results = []
        for lb, w in zip([22, 63, 126, 252], mom_w):
            if len(h) >= lb + 5:
                results.append(w * float(h.iloc[-1]/h.iloc[-lb]-1))
            else:
                results.append(None)
        if any(r is None for r in results): return np.nan
        return sum(results)

    monthly_weights = {}
    processed = set()
    btc_dd_blocked = False
    vol_scale = 1.0
    prev_w = {}

    val = 1.0; peak = 1.0
    equity_vals, equity_dates = [], []

    for day_idx, day in enumerate(trading_days):
        # 第一天没有 prev_day，跳过（无法计算收益）
        if day_idx == 0:
            equity_vals.append(val); equity_dates.append(day); continue
        # 修复前瞻偏差：所有日频信号必须用昨天收盘价（prev_day），不能用当天收盘价
        prev_day = trading_days[day_idx - 1]

        # 月末更新信号
        past_mes = month_ends[month_ends < day]
        if len(past_mes) > 0:
            last_me = past_mes[-1]
            if last_me not in processed:
                btc_m = multi_mom(btc_p, last_me)
                eth_m = multi_mom(eth_p, last_me)

                # 各自 MA 状态
                btc_above = (not np.isnan(btc_m)) and \
                    len(btc_ma.loc[:last_me].dropna()) > 0 and \
                    float(btc_p.loc[:last_me].dropna().iloc[-1]) > \
                    float(btc_ma.loc[:last_me].dropna().iloc[-1])
                eth_above = (not np.isnan(eth_m)) and \
                    len(eth_ma.loc[:last_me].dropna()) > 0 and \
                    float(eth_p.loc[:last_me].dropna().iloc[-1]) > \
                    float(eth_ma.loc[:last_me].dropna().iloc[-1])

                # 计算本月目标权重（月频信号层）
                if strategy == 'v3':
                    # 复现 v3：赢家拿70%，BTC-MA驱动
                    if btc_above or eth_above:
                        if (not np.isnan(btc_m)) and (not np.isnan(eth_m)) and btc_m >= eth_m:
                            mw = {'BTC': 0.70, 'ETH': 0.30}
                        else:
                            mw = {'ETH': 0.70, 'BTC': 0.30}
                    else:
                        mw = {'GLD': 1.0}

                elif strategy == 'A':
                    # 连续动量比例，各自独立MA（月末决策）
                    btc_w, eth_w = softmax_split(
                        btc_m if btc_above else -999,
                        eth_m if eth_above else -999)
                    mw = {}
                    if btc_above: mw['BTC'] = btc_w
                    if eth_above: mw['ETH'] = eth_w
                    if not mw: mw = {'GLD': 1.0}

                elif strategy == 'B':
                    # 各自独立MA + 50:50
                    mw = {}
                    if btc_above: mw['BTC'] = 0.5
                    if eth_above: mw['ETH'] = 0.5
                    if not mw: mw = {'GLD': 1.0}
                    # 重新归一化
                    if mw and 'GLD' not in mw:
                        total = sum(mw.values())
                        mw = {k: v/total for k,v in mw.items()}

                elif strategy == 'C':
                    # 连续权重 + 各自独立MA ← 核心
                    if btc_above and eth_above:
                        btc_w, eth_w = softmax_split(btc_m, eth_m)
                        mw = {'BTC': btc_w, 'ETH': eth_w}
                    elif btc_above:
                        mw = {'BTC': 1.0}
                    elif eth_above:
                        mw = {'ETH': 1.0}
                    else:
                        mw = {'GLD': 1.0}

                elif strategy == 'D':
                    # 阈值切换 + 独立MA
                    btc_m_safe = btc_m if (not np.isnan(btc_m)) else -1
                    eth_m_safe = eth_m if (not np.isnan(eth_m)) else -1
                    if btc_above and eth_above:
                        diff = abs(btc_m_safe - eth_m_safe)
                        if diff > MOM_SWITCH_THRESH:
                            # 强势者拿80%
                            if btc_m_safe > eth_m_safe:
                                mw = {'BTC': 0.80, 'ETH': 0.20}
                            else:
                                mw = {'ETH': 0.80, 'BTC': 0.20}
                        else:
                            mw = {'BTC': 0.50, 'ETH': 0.50}
                    elif btc_above:
                        mw = {'BTC': 1.0}
                    elif eth_above:
                        mw = {'ETH': 1.0}
                    else:
                        mw = {'GLD': 1.0}
                else:
                    mw = {'BTC': 0.5, 'ETH': 0.5}

                # vol 目标化（月频）
                bv = btc_vol.loc[:last_me].dropna()
                vol_scale = float(np.clip(
                    CRYPTO_VOL_TGT / max(float(bv.iloc[-1]), 0.01),
                    CRYPTO_VOL_MIN, CRYPTO_VOL_MAX)) if len(bv) > 0 else 1.0

                monthly_weights[last_me] = mw
                processed.add(last_me)

        # 日频保护（实时 BTC + ETH MA）
        app_mes = [me for me in month_ends if me < day and me in monthly_weights]
        if not app_mes:
            equity_vals.append(val); equity_dates.append(day); continue
        cur_me = app_mes[-1]
        base_w = monthly_weights[cur_me].copy()

        # 日频 BTC/ETH MA 过滤（必须用 prev_day 收盘价，避免前瞻偏差）
        btc_now = btc_p.loc[:prev_day].dropna()
        btc_ma_now = btc_ma.loc[:prev_day].dropna()
        eth_now = eth_p.loc[:prev_day].dropna()
        eth_ma_now = eth_ma.loc[:prev_day].dropna()

        btc_daily_above = (len(btc_now) > 0 and len(btc_ma_now) > 0 and
                           float(btc_now.iloc[-1]) > float(btc_ma_now.iloc[-1]))
        eth_daily_above = (len(eth_now) > 0 and len(eth_ma_now) > 0 and
                           float(eth_now.iloc[-1]) > float(eth_ma_now.iloc[-1]))

        # BTC 回撤保护（基于 prev_day，避免前瞻偏差）
        if len(btc_now) >= 5:
            btc_peak_val = float(btc_p.loc[:prev_day].max())
            btc_dd = float(btc_now.iloc[-1] / btc_peak_val - 1)
            if btc_dd < CRYPTO_DD_EXIT: btc_dd_blocked = True
            elif btc_dd > CRYPTO_DD_REENTRY: btc_dd_blocked = False

        # 构建当天实际权重
        if btc_dd_blocked:
            # 极端情况：全部退出到 GLD
            current_w = {'GLD': 1.0}
        elif strategy == 'v3':
            # v3 逻辑：BTC MA 控制全局
            if not btc_daily_above and 'GLD' not in base_w:
                current_w = {'GLD': 1.0}
            else:
                current_w = base_w.copy()
        else:
            # v5 策略：各自独立 MA（日频）
            current_w = {}
            for coin, w in base_w.items():
                if coin == 'GLD':
                    current_w['GLD'] = current_w.get('GLD', 0) + w
                elif coin == 'BTC':
                    if btc_daily_above:
                        current_w['BTC'] = w * vol_scale
                        gld_supp = w * max(1 - vol_scale, 0)
                        if gld_supp > 0.01:
                            current_w['GLD'] = current_w.get('GLD', 0) + gld_supp
                    else:
                        current_w['GLD'] = current_w.get('GLD', 0) + w
                elif coin == 'ETH':
                    if eth_daily_above:
                        current_w['ETH'] = w * vol_scale
                        gld_supp = w * max(1 - vol_scale, 0)
                        if gld_supp > 0.01:
                            current_w['GLD'] = current_w.get('GLD', 0) + gld_supp
                    else:
                        current_w['GLD'] = current_w.get('GLD', 0) + w

            if not current_w:
                current_w = {'GLD': 1.0}

        if day_idx == 0:
            equity_vals.append(val); equity_dates.append(day); continue

        # 换仓成本
        app_mes2 = [me for me in month_ends if me < day and me in monthly_weights]
        if len(app_mes2) >= 2:
            nxt = trading_days[trading_days > app_mes2[-1]]
            if len(nxt) > 0 and day == nxt[0]:
                tc = sum(abs(current_w.get(k,0) - prev_w.get(k,0))
                         for k in set(current_w)|set(prev_w)) * COST
                val *= (1 - tc)
        prev_w = current_w.copy()

        # prev_day 已在循环头部定义
        day_ret = 0.0
        for ticker, w in current_w.items():
            if w == 0: continue
            series = btc_p if ticker=='BTC' else eth_p if ticker=='ETH' else gld_p
            if prev_day in series.index and day in series.index:
                p0, p1 = series.loc[prev_day], series.loc[day]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    day_ret += (p1/p0 - 1) * w

        val *= (1 + day_ret)
        peak = max(peak, val)
        equity_vals.append(val); equity_dates.append(day)

    return pd.Series(equity_vals, index=equity_dates)

def calc_metrics(equity, start='2015-01-01', end='2025-12-31'):
    eq = equity.loc[start:end].dropna()
    if len(eq) < 100: return None
    rets = eq.pct_change().dropna().values
    years = len(rets) / 252
    cagr  = float((eq.iloc[-1]/eq.iloc[0])**(1/years) - 1)
    cum   = np.cumprod(1 + rets)
    max_dd = float(((cum - np.maximum.accumulate(cum))/np.maximum.accumulate(cum)).min())
    excess = rets - 0.04/252
    sharpe = float(np.mean(excess)/np.std(excess)*np.sqrt(252)) if np.std(excess) > 0 else 0
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0

    def ss(s, e):
        sub = equity.loc[s:e].dropna()
        if len(sub) < 50: return 0.0
        r = sub.pct_change().dropna().values
        ex = r - 0.04/252
        return float(np.mean(ex)/np.std(ex)*np.sqrt(252)) if np.std(ex) > 0 else 0.0
    def sc(s, e):
        sub = equity.loc[s:e].dropna()
        if len(sub) < 50: return 0.0
        y = len(sub)/252
        return float((sub.iloc[-1]/sub.iloc[0])**(1/y)-1)

    # 与 v3 保持一致：IS=2015-2020, OOS=2021-2025
    is_s = ss('2015-01-01','2020-12-31')
    oos_s = ss('2021-01-01','2025-12-31')
    wf = float(oos_s/is_s) if is_s > 0 else 0.0
    comp = sharpe*0.25 + calmar*0.25 + min(cagr, 2.0)*0.50
    return dict(cagr=cagr, max_dd=max_dd, sharpe=sharpe, calmar=calmar,
                wf=wf, is_sharpe=is_s, oos_sharpe=oos_s,
                is_cagr=sc('2015-01-01','2020-12-31'),
                oos_cagr=sc('2021-01-01','2025-12-31'), composite=comp)

def main():
    print("="*72)
    print("🐻 Hybrid v5 — 动态 BTC/ETH 权重优化")
    print("="*72)

    btc_p = load_series('BTC_USD')
    eth_p = load_series('ETH_USD')
    gld_p = load_series('GLD')

    # 加载 v3 stock equity（直接复用）
    v3_eq = pd.read_csv(BASE/'hybrid/codebear/hybrid_v3_equity.csv',
                         index_col=0, parse_dates=True)
    stock_eq = v3_eq['StockV9j']
    print(f"  Stock (v9j): {stock_eq.index[0].date()} → {stock_eq.index[-1].date()}")

    strategies = ['v3', 'A', 'B', 'C', 'D']
    strat_names = {
        'v3': 'v3复现（基准）',
        'A':  '连续动量权重',
        'B':  '独立MA+等权',
        'C':  '连续权重+独立MA ★',
        'D':  '阈值切换+独立MA',
    }
    w_crypto_vals = [0.15, 0.20, 0.25, 0.30]

    all_results = []

    # 先跑各策略 Crypto 独立指标
    print("\n📊 Crypto 组件独立对比:")
    print(f"{'策略':>20} | {'CAGR':>8} {'MaxDD':>7} {'Sharpe':>7} {'WF':>6} "
          f"{'IS_CAGR':>8} {'OOS_CAGR':>8}")
    print("-"*72)

    crypto_equities = {}
    for strat in strategies:
        ceq = run_crypto_v5_daily(btc_p, eth_p, gld_p, strategy=strat)
        crypto_equities[strat] = ceq
        m = calc_metrics(ceq)
        if m:
            flag = "★" if strat == 'C' else " "
            print(f"  {flag} {strat_names[strat]:>18} | "
                  f"{m['cagr']:>7.1%} {m['max_dd']:>7.1%} "
                  f"{m['sharpe']:>7.2f} {m['wf']:>6.3f} "
                  f"{m['is_cagr']:>7.1%} {m['oos_cagr']:>8.1%}")

    # 组合扫描
    print("\n🔍 组合扫描 (目标: WF≥0.58, CAGR≥50%):")
    print(f"{'策略':>14} {'w_c':>4} | {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} "
          f"{'WF':>6} {'IS_S':>6} {'OOS_S':>6} {'Comp':>7}")
    print("-"*76)

    idx_full = stock_eq.index.intersection(
        list(crypto_equities.values())[0].index)
    idx_full = idx_full[(idx_full>='2015-01-01') & (idx_full<='2025-12-31')]
    s_ret = stock_eq.loc[idx_full].pct_change().fillna(0)

    best_per_strat = {}
    for strat in strategies:
        ceq = crypto_equities[strat]
        idx = stock_eq.index.intersection(ceq.index)
        idx = idx[(idx>='2015-01-01') & (idx<='2025-12-31')]
        s_r = stock_eq.loc[idx].pct_change().fillna(0)
        c_r = ceq.loc[idx].pct_change().fillna(0)

        best_wf = None
        for wc in w_crypto_vals:
            combo = (1+((1-wc)*s_r + wc*c_r)).cumprod()
            m = calc_metrics(combo)
            if m is None: continue
            flag = "🏆" if m['wf']>=0.60 and m['cagr']>=0.50 else \
                   ("✅" if m['wf']>=0.58 else "  ")
            print(f"  {flag} {strat_names[strat][:12]:>12} {wc:.0%} | "
                  f"{m['cagr']:>6.1%} {m['max_dd']:>7.1%} "
                  f"{m['sharpe']:>7.2f} {m['wf']:>6.3f} "
                  f"{m['is_sharpe']:>6.2f} {m['oos_sharpe']:>6.2f} {m['composite']:>7.3f}")
            r = dict(strategy=strat, w_crypto=wc, **m)
            all_results.append(r)
            if m['wf'] >= 0.58:
                if best_wf is None or m['composite'] > best_wf['composite']:
                    best_wf = r
        if best_wf:
            best_per_strat[strat] = best_wf
        print()

    # 汇总
    print("="*72)
    print("📊 各策略 WF≥0.58 最优配置:")
    v3_ref = next((r for r in all_results
                   if r['strategy']=='v3' and abs(r['w_crypto']-0.20)<0.01), None)
    for strat in strategies:
        b = best_per_strat.get(strat)
        if b:
            delta_c = b['cagr'] - (v3_ref['cagr'] if v3_ref else 0)
            delta_w = b['wf']   - (v3_ref['wf']   if v3_ref else 0)
            flag = "🏆" if b['wf']>=(v3_ref['wf'] if v3_ref else 0.6) else "✅"
            print(f"  {flag} {strat_names[strat]}: w={b['w_crypto']:.0%} "
                  f"CAGR={b['cagr']:.1%}(Δ{delta_c:+.1%}) "
                  f"WF={b['wf']:.3f}(Δ{delta_w:+.3f}) MaxDD={b['max_dd']:.1%}")
        else:
            print(f"  ❌ {strat_names[strat]}: 无 WF≥0.58 配置")

    # 找全局最优
    wf_ok = [r for r in all_results if r['wf'] >= 0.58]
    if wf_ok:
        best = max(wf_ok, key=lambda x: x['composite'])
        print(f"\n🏆 全局最优: {strat_names[best['strategy']]} w={best['w_crypto']:.0%}")
        print(f"   CAGR={best['cagr']:.1%} WF={best['wf']:.3f} "
              f"MaxDD={best['max_dd']:.1%} Sharpe={best['sharpe']:.2f} Comp={best['composite']:.3f}")
        if v3_ref:
            print(f"   vs v3(w=20%): CAGR{best['cagr']-v3_ref['cagr']:+.1%} "
                  f"WF{best['wf']-v3_ref['wf']:+.3f}")

    # 保存
    out = {
        'strategy': 'Hybrid v5: Dynamic BTC/ETH Weight Optimization',
        'crypto_components': {s: calc_metrics(crypto_equities[s]) for s in strategies},
        'sweep': all_results,
        'best_wf058': max(wf_ok, key=lambda x:x['composite']) if wf_ok else None,
        'meta': {'date':'2026-02-22', 'author':'代码熊 🐻',
                 'note':'v3 stock + improved BTC/ETH rotation'}
    }
    json.dump(out, open(Path(__file__).with_name('hybrid_v5_results.json'),'w'),
              indent=2, default=float)
    print(f"\n💾 结果保存 → hybrid_v5_results.json")

if __name__ == '__main__':
    main()
