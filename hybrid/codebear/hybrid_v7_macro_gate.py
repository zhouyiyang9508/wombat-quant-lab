#!/usr/bin/env python3
"""
Hybrid v7 — v11b 股票 + Crypto v5-D + 宏观多信号门控
代码熊 🐻 | 2026-02-22

Direction E: 在 Crypto v5-D 基础上叠加宏观门控：
  Layer 1 (已有): BTC/ETH 各自 200dMA 独立过滤
  Layer 2 (已有): BTC DD 迟滞保护 (-25% exit, -15% reentry)
  Layer 3 (已有): BTC vol targeting (60d vol → scale crypto weight)
  Layer 4 (新): SPY 已实现波动率门控 (VIX proxy)
    - SPY vol30 > 0.30 → crypto 仓位全部→GLD (极端恐慌)
    - SPY vol30 > 0.22 → crypto 仓位减半→GLD (高度恐慌)
  Layer 5 (新): HYG 信用利差门控
    - HYG < 200dMA → crypto 仓位减半→GLD
  Layer 6 (新): 双重确认（SPY vol高 + HYG信用压力同时）→ 全退出

注：这些宏观门控在纯股票端无效（v16已验证），但对crypto端可能有效：
  - 2022年初：VIX升高+HYG跌破200dMA+BTC跌破200dMA，三重确认
  - 如果BTC 200dMA滞后了，VIX和HYG可以提前触发，减少OOS损失→提升WF

评估：
  - IS=2015-2020, OOS=2021-2025
  - Composite = Sharpe×0.4 + Calmar×0.4 + min(CAGR,1.0)×0.2
  - 目标: WF>0.644 (超越v6), CAGR>45%

同时测试 Direction D (跨资产动量):
  简化版: 用SPY/QQQ/IWM/EFA/EEM动量选前2只→在对应国内/国外权重中分配
"""

import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parents[2]
CACHE = BASE / "data_cache"

sys.path.insert(0, str(BASE))
from hybrid.codebear.hybrid_v5_dynbtceth import load_series, calc_metrics

# Crypto 参数
CRYPTO_MA_WIN     = 200
CRYPTO_DD_EXIT    = -0.25
CRYPTO_DD_REENTRY = -0.15
CRYPTO_VOL_WIN    = 60
CRYPTO_VOL_TGT    = 0.80
CRYPTO_VOL_MIN    = 0.40
CRYPTO_VOL_MAX    = 1.50
CRYPTO_MOM_LB     = 90
MOM_SWITCH_THRESH = 0.05
COST = 0.0015

# 宏观门控参数
SPY_VOL_FULL_EXIT = 0.30     # SPY vol30 > 0.30 → crypto全退出
SPY_VOL_HALF_EXIT = 0.22     # SPY vol30 > 0.22 → crypto减半
HYG_MA_WINDOW     = 200      # HYG 200dMA
HYG_CRYPTO_CUT    = 0.50     # HYG < 200dMA → crypto仓位×0.50


def walk_forward(equity, is_end='2020-12-31', oos_start='2021-01-01', rf=0.04):
    eq = equity.dropna()
    is_eq  = eq[eq.index <= pd.Timestamp(is_end)]
    oos_eq = eq[eq.index >= pd.Timestamp(oos_start)]
    if len(is_eq) < 30 or len(oos_eq) < 30:
        return dict(wf=0, is_sh=0, oos_sh=0, is_cagr=0, oos_cagr=0)
    def sh(e):
        r = e.pct_change().dropna().values
        ex = r - rf/252
        return float(np.mean(ex)/np.std(ex)*np.sqrt(252)) if np.std(ex) > 0 else 0.0
    def cg(e):
        y = len(e)/252
        return float((e.iloc[-1]/e.iloc[0])**(1/y)-1) if y > 0 else 0.0
    is_s = sh(is_eq); oos_s = sh(oos_eq)
    wf = oos_s / is_s if is_s > 0 else 0.0
    return dict(wf=round(wf,3), is_sh=round(is_s,3), oos_sh=round(oos_s,3),
                is_cagr=round(cg(is_eq),4), oos_cagr=round(cg(oos_eq),4))


def build_hybrid(stock_eq, crypto_eq, w_crypto):
    idx = stock_eq.index.intersection(crypto_eq.index).sort_values()
    s_r = stock_eq.loc[idx].pct_change().fillna(0)
    c_r = crypto_eq.loc[idx].pct_change().fillna(0)
    combo = (1 + (1-w_crypto)*s_r + w_crypto*c_r).cumprod()
    return combo


def run_crypto_v7_daily(btc_p, eth_p, gld_p, spy_close, hyg_p,
                         spy_vol_full=SPY_VOL_FULL_EXIT,
                         spy_vol_half=SPY_VOL_HALF_EXIT,
                         hyg_cut=HYG_CRYPTO_CUT,
                         start='2015-01-01', end='2025-12-31'):
    """
    Crypto v7: v5-D 策略 + 宏观多信号门控
    """
    mom_w = (0.20, 0.50, 0.20, 0.10)

    idx = btc_p.loc[start:end].dropna().index.union(
          eth_p.loc[start:end].dropna().index)
    trading_days = idx
    month_ends = pd.Series(1, index=trading_days).resample('ME').last().index

    # 预计算
    btc_ma  = btc_p.rolling(CRYPTO_MA_WIN).mean()
    eth_ma  = eth_p.rolling(CRYPTO_MA_WIN).mean()
    btc_vol = np.log(btc_p/btc_p.shift(1)).rolling(CRYPTO_VOL_WIN).std()*np.sqrt(252)

    # SPY vol30 (VIX proxy)
    spy_log_ret = np.log(spy_close/spy_close.shift(1))
    spy_vol30 = spy_log_ret.rolling(30, min_periods=20).std() * np.sqrt(252)

    # HYG 200dMA
    hyg_ma = hyg_p.rolling(HYG_MA_WINDOW, min_periods=int(HYG_MA_WINDOW*0.8)).mean()

    def multi_mom(prices, date):
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
    macro_log = []

    val = 1.0
    equity_vals, equity_dates = [], []

    for day_idx, day in enumerate(trading_days):
        if day_idx == 0:
            equity_vals.append(val); equity_dates.append(day); continue
        prev_day = trading_days[day_idx - 1]

        # 月末更新信号
        past_mes = month_ends[month_ends < day]
        if len(past_mes) > 0:
            last_me = past_mes[-1]
            if last_me not in processed:
                btc_m = multi_mom(btc_p, last_me)
                eth_m = multi_mom(eth_p, last_me)

                btc_above = (not np.isnan(btc_m)) and \
                    len(btc_ma.loc[:last_me].dropna()) > 0 and \
                    float(btc_p.loc[:last_me].dropna().iloc[-1]) > \
                    float(btc_ma.loc[:last_me].dropna().iloc[-1])
                eth_above = (not np.isnan(eth_m)) and \
                    len(eth_ma.loc[:last_me].dropna()) > 0 and \
                    float(eth_p.loc[:last_me].dropna().iloc[-1]) > \
                    float(eth_ma.loc[:last_me].dropna().iloc[-1])

                # Strategy D: 阈值切换 + 独立MA
                btc_m_safe = btc_m if (not np.isnan(btc_m)) else -1
                eth_m_safe = eth_m if (not np.isnan(eth_m)) else -1
                if btc_above and eth_above:
                    diff = abs(btc_m_safe - eth_m_safe)
                    if diff > MOM_SWITCH_THRESH:
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

                # vol 目标化
                bv = btc_vol.loc[:last_me].dropna()
                vol_scale = float(np.clip(
                    CRYPTO_VOL_TGT / max(float(bv.iloc[-1]), 0.01),
                    CRYPTO_VOL_MIN, CRYPTO_VOL_MAX)) if len(bv) > 0 else 1.0

                monthly_weights[last_me] = mw
                processed.add(last_me)

        # 日频保护层
        app_mes = [me for me in month_ends if me < day and me in monthly_weights]
        if not app_mes:
            equity_vals.append(val); equity_dates.append(day); continue
        cur_me = app_mes[-1]
        base_w = monthly_weights[cur_me].copy()

        # MA过滤 (prev_day)
        btc_now = btc_p.loc[:prev_day].dropna()
        btc_ma_now = btc_ma.loc[:prev_day].dropna()
        eth_now = eth_p.loc[:prev_day].dropna()
        eth_ma_now = eth_ma.loc[:prev_day].dropna()

        btc_daily_above = (len(btc_now) > 0 and len(btc_ma_now) > 0 and
                           float(btc_now.iloc[-1]) > float(btc_ma_now.iloc[-1]))
        eth_daily_above = (len(eth_now) > 0 and len(eth_ma_now) > 0 and
                           float(eth_now.iloc[-1]) > float(eth_ma_now.iloc[-1]))

        # BTC DD 保护 (prev_day)
        if len(btc_now) >= 5:
            btc_peak_val = float(btc_p.loc[:prev_day].max())
            btc_dd = float(btc_now.iloc[-1] / btc_peak_val - 1)
            if btc_dd < CRYPTO_DD_EXIT: btc_dd_blocked = True
            elif btc_dd > CRYPTO_DD_REENTRY: btc_dd_blocked = False

        # ★ 新 Layer 4: SPY vol30 宏观门控 (prev_day) ★
        spy_v = spy_vol30.loc[:prev_day].dropna()
        cur_spy_vol = float(spy_v.iloc[-1]) if len(spy_v) > 0 else 0.15
        macro_full_exit = (cur_spy_vol >= spy_vol_full)
        macro_half_exit = (cur_spy_vol >= spy_vol_half and not macro_full_exit)

        # ★ 新 Layer 5: HYG 信用利差门控 (prev_day) ★
        hyg_now = hyg_p.loc[:prev_day].dropna()
        hyg_ma_now = hyg_ma.loc[:prev_day].dropna()
        hyg_stressed = False
        if len(hyg_now) > 0 and len(hyg_ma_now) > 0:
            hyg_stressed = bool(float(hyg_now.iloc[-1]) < float(hyg_ma_now.iloc[-1]))

        # ★ 新 Layer 6: 双重确认 ★
        double_stress = macro_half_exit and hyg_stressed

        # 构建当天实际权重
        if btc_dd_blocked or macro_full_exit:
            current_w = {'GLD': 1.0}
        else:
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

            # 应用宏观门控缩减
            if double_stress:
                # 双重确认 → 全退出到GLD
                crypto_frac = sum(current_w.get(c, 0) for c in ('BTC', 'ETH'))
                if crypto_frac > 0:
                    current_w['GLD'] = current_w.get('GLD', 0) + crypto_frac
                    current_w.pop('BTC', None)
                    current_w.pop('ETH', None)
            elif macro_half_exit:
                # SPY高vol → crypto减半
                for c in ('BTC', 'ETH'):
                    if c in current_w:
                        cut = current_w[c] * 0.5
                        current_w[c] -= cut
                        current_w['GLD'] = current_w.get('GLD', 0) + cut
                        if current_w[c] < 0.01: current_w.pop(c)
            elif hyg_stressed:
                # HYG信用压力 → crypto×hyg_cut
                for c in ('BTC', 'ETH'):
                    if c in current_w:
                        cut = current_w[c] * (1 - hyg_cut)
                        current_w[c] *= hyg_cut
                        current_w['GLD'] = current_w.get('GLD', 0) + cut
                        if current_w[c] < 0.01: current_w.pop(c)

        # 换仓成本
        app_mes2 = [me for me in month_ends if me < day and me in monthly_weights]
        if len(app_mes2) >= 2:
            nxt = trading_days[trading_days > app_mes2[-1]]
            if len(nxt) > 0 and day == nxt[0]:
                tc = sum(abs(current_w.get(k,0) - prev_w.get(k,0))
                         for k in set(current_w)|set(prev_w)) * COST
                val *= (1 - tc)

        # 日频收益
        day_ret = 0.0
        for asset, w in current_w.items():
            series = {'BTC': btc_p, 'ETH': eth_p, 'GLD': gld_p}.get(asset)
            if series is None: continue
            if prev_day in series.index and day in series.index:
                p0 = series[prev_day]; p1 = series[day]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    day_ret += (p1/p0 - 1) * w

        val *= (1 + day_ret)
        equity_vals.append(val); equity_dates.append(day)
        prev_w = current_w.copy()

    return pd.Series(equity_vals, index=equity_dates, name='CryptoV7')


def main():
    print("=" * 78)
    print("🐻 Hybrid v7 — v11b + Crypto v5-D + 宏观多信号门控 (Direction E)")
    print("新增: SPY vol30(VIX proxy) + HYG信用利差 → crypto 仓位管理")
    print("=" * 78)

    # 加载数据
    print("\n[1/4] 加载数据...")
    v4_eq = pd.read_csv(BASE/'hybrid/codebear/hybrid_v4_equity.csv',
                         index_col=0, parse_dates=True)
    stock_eq = v4_eq['StockV11b'].dropna()
    print(f"  StockV11b: {len(stock_eq)} 日")

    btc_p = load_series('BTC_USD')
    eth_p = load_series('ETH_USD')
    gld_p = load_series('GLD')

    spy_df = pd.read_csv(CACHE/'stocks/SPY.csv')
    c = 'Date' if 'Date' in spy_df.columns else spy_df.columns[0]
    spy_df[c] = pd.to_datetime(spy_df[c])
    spy_df = spy_df.set_index(c).sort_index()
    spy_close = pd.to_numeric(spy_df['Close'], errors='coerce').dropna()

    hyg_p = load_series('HYG')
    print(f"  HYG: {len(hyg_p)} 日 ({hyg_p.index[0].date()} → {hyg_p.index[-1].date()})")
    print(f"  SPY: {len(spy_close)} 日")

    # 计算 v11b baseline
    sm = calc_metrics(stock_eq)
    swf = walk_forward(stock_eq)
    print(f"  Stock v11b: CAGR={sm['cagr']:.1%}, MaxDD={sm['max_dd']:.1%}, "
          f"Sharpe={sm['sharpe']:.2f}, WF={swf['wf']:.3f}")

    # [2/4] Crypto v5-D baseline (无宏观门控)
    print("\n[2/4] Crypto v5-D baseline...")
    from hybrid_v5_dynbtceth import run_crypto_v5_daily
    crypto_v5d = run_crypto_v5_daily(btc_p, eth_p, gld_p, strategy='D')
    cm5 = calc_metrics(crypto_v5d)
    cwf5 = walk_forward(crypto_v5d)
    print(f"  Crypto v5-D: CAGR={cm5['cagr']:.1%}, MaxDD={cm5['max_dd']:.1%}, "
          f"Sharpe={cm5['sharpe']:.2f}, WF={cwf5['wf']:.3f}")

    # [3/4] Crypto v7 (宏观门控)
    print("\n[3/4] Crypto v7 宏观门控扫描...")

    # 扫描不同宏观门控配置
    configs = [
        ('v5-D baseline',       999, 999, 1.0),    # 门控关闭（无穷阈值，不触发）
        ('SPY_vol only(0.30)',  0.30, 999, 1.0),    # 仅 SPY vol 全退出
        ('SPY_vol only(0.25)',  0.25, 999, 1.0),
        ('HYG only(50%)',       999, 999, 0.50),    # 仅 HYG 门控
        ('HYG only(30%)',       999, 999, 0.30),
        ('SPY+HYG light',      0.30, 0.25, 0.60),  # 轻度组合
        ('SPY+HYG medium',     0.30, 0.22, 0.50),  # 中度组合（默认）
        ('SPY+HYG tight',      0.25, 0.20, 0.40),  # 紧缩组合
        ('SPY+HYG very tight', 0.25, 0.18, 0.30),  # 极端紧缩
    ]

    print(f"\n{'配置':<25} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'WF':>7} "
          f"{'IS_Sh':>7} {'OOS_Sh':>7} {'OOS_CAG':>8}")
    print("-" * 90)

    crypto_results = {}
    for label, spy_full, spy_half, hyg_c in configs:
        c_eq = run_crypto_v7_daily(btc_p, eth_p, gld_p, spy_close, hyg_p,
                                    spy_vol_full=spy_full, spy_vol_half=spy_half,
                                    hyg_cut=hyg_c)
        cm = calc_metrics(c_eq)
        cw = walk_forward(c_eq)
        crypto_results[label] = (c_eq, cm, cw)
        wf_flag = "✅" if cw['wf'] >= 0.60 else ("⚠️" if cw['wf'] >= 0.55 else "❌")
        print(f"  {label:<23} {cm['cagr']:>7.1%}  {cm['max_dd']:>7.1%}  "
              f"{cm['sharpe']:>7.2f}  {wf_flag}{cw['wf']:.2f}  "
              f"{cw['is_sh']:>6.2f}  {cw['oos_sh']:>6.2f}  {cw.get('oos_cagr',0):>7.1%}")

    # [4/4] Hybrid 组合扫描 (所有crypto配置 × w_crypto)
    print("\n[4/4] Hybrid v7 组合扫描...")
    w_list = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

    print(f"\n{'Crypto配置':<25} {'w_c':>5} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Comp':>8} "
          f"{'WF':>7} {'IS_Sh':>7} {'OOS_Sh':>7}")
    print("-" * 105)

    best_overall = None
    best_comp = -99

    for label, (c_eq, cm, cw) in crypto_results.items():
        for w_c in w_list:
            h_eq = build_hybrid(stock_eq, c_eq, w_c)
            hm = calc_metrics(h_eq)
            hw = walk_forward(h_eq)
            comp = hm['sharpe']*0.4 + hm['calmar']*0.4 + min(hm['cagr'],1.0)*0.2
            wf_flag = "✅" if hw['wf'] >= 0.60 else ("⚠️" if hw['wf'] >= 0.55 else "❌")

            # 只显示重要结果（WF>=0.55 或 w_c=0.15）
            if hw['wf'] >= 0.55 or w_c == 0.15:
                print(f"  {label:<23} {w_c:>4.0%}  {hm['cagr']:>7.1%}  {hm['max_dd']:>7.1%}  "
                      f"{hm['sharpe']:>7.2f}  {comp:>7.3f}  {wf_flag}{hw['wf']:.2f}  "
                      f"{hw['is_sh']:>6.2f}  {hw['oos_sh']:>6.2f}")

            if hw['wf'] >= 0.60 and comp > best_comp:
                best_comp = comp
                best_overall = dict(
                    config=label, w_crypto=w_c,
                    cagr=hm['cagr'], max_dd=hm['max_dd'],
                    sharpe=hm['sharpe'], calmar=hm['calmar'],
                    composite=comp, **hw
                )

    # Hybrid v6 baseline reference
    print()
    v6_ref = dict(cagr=0.452, max_dd=-0.189, sharpe=2.065, composite=1.382,
                  wf=0.644, is_sh=2.477, oos_sh=1.595)
    print(f"  {'Hybrid v6 REF (15%)':23}  15%  {v6_ref['cagr']:>7.1%}  {v6_ref['max_dd']:>7.1%}  "
          f"{v6_ref['sharpe']:>7.2f}  {v6_ref['composite']:>7.3f}  ✅{v6_ref['wf']:.2f}  "
          f"{v6_ref['is_sh']:>6.2f}  {v6_ref['oos_sh']:>6.2f}  ← v6冠军")
    print("=" * 105)

    # 结论
    if best_overall:
        print(f"\n🏆 Hybrid v7 最优（WF≥0.60）：{best_overall['config']}  w_crypto={best_overall['w_crypto']:.0%}")
        print(f"   CAGR={best_overall['cagr']:.1%} / MaxDD={best_overall['max_dd']:.1%} / "
              f"WF={best_overall['wf']:.3f} / Sharpe={best_overall['sharpe']:.3f} / "
              f"Composite={best_overall['composite']:.3f}")
        cagr_d = best_overall['cagr'] - v6_ref['cagr']
        wf_d = best_overall['wf'] - v6_ref['wf']
        comp_d = best_overall['composite'] - v6_ref['composite']
        print(f"   vs v6: CAGR Δ{cagr_d:+.1%}  WF Δ{wf_d:+.3f}  Composite Δ{comp_d:+.3f}")

        if best_overall['cagr'] > 0.48 and best_overall['wf'] > 0.62:
            print("\n🚀🚀🚀 【重大突破】CAGR>48% + WF>0.62！")
        elif best_overall['wf'] > v6_ref['wf'] + 0.01:
            print(f"\n✅ WF 改善: {v6_ref['wf']:.3f} → {best_overall['wf']:.3f}")
        else:
            print("\n⚠️  未超越 v6")
    else:
        print("\n⚠️  没有 WF≥0.60 的配置")

    # 保存
    out = {
        'v6_reference': v6_ref,
        'best_v7': best_overall,
        'crypto_results': {k: {'metrics': v[1], 'wf': v[2]} for k, v in crypto_results.items()},
        'meta': {
            'date': '2026-02-22',
            'desc': 'Direction E: Macro gating on crypto (SPY vol30 + HYG credit spread)'
        }
    }
    jf = Path(__file__).parent / "hybrid_v7_results.json"
    jf.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n💾 → {jf}")

    # Save equity if best found
    if best_overall and best_overall['w_crypto'] > 0:
        label = best_overall['config']
        c_eq = crypto_results[label][0]
        best_eq = build_hybrid(stock_eq, c_eq, best_overall['w_crypto'])
        eq_df = pd.DataFrame({
            'StockV11b': stock_eq,
            'CryptoV7_best': c_eq,
            'HybridV7_best': best_eq
        }).dropna()
        eq_df.to_csv(Path(__file__).parent / "hybrid_v7_equity.csv")
        print(f"📁 净值曲线已保存")

    return out


if __name__ == '__main__':
    main()
