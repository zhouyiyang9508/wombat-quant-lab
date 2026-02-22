#!/usr/bin/env python3
"""
Hybrid v8 — 三组件混合：v11b股票 + Crypto v5-D + ETF宏观动量
代码熊 🐻 | 2026-02-22

Direction D 变体：跨资产动量作为第三独立组件

架构创新：
  组件1: v11b 股票 (S&P500 动量轮换 + 三层防御)
  组件2: Crypto v5-D (BTC/ETH 独立200dMA + DD + vol targeting)
  组件3: ETF 宏观动量 (新！)
    - 资产池：QQQ, SPY, IWM, EFA, EEM, GLD, TLT
    - 月频信号：6m动量排名→选Top2 ETF（但如果Top2动量都<=0→全SHY）
    - 趋势过滤：只买入 price > 200dMA 的 ETF
    - 与v11b不同：v11b是行业内选股，ETF组件是跨市场宏观配置

假设：
  - 三组件低相关：v11b选个股, ETF选宏观方向, Crypto独立
  - ETF动量在2022年会快速切到GLD/TLT/SHY（避险）
  - OOS 期间 ETF 组件可能更稳定（更少噪音）

权重方案扫描：
  w_stock:w_etf:w_crypto = [各种比例]
  例如 60:20:20, 70:15:15, 50:30:20

评估：IS=2015-2020, OOS=2021-2025, Composite公式同前
"""

import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parents[2]
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

sys.path.insert(0, str(BASE))
from hybrid.codebear.hybrid_v5_dynbtceth import load_series, run_crypto_v5_daily, calc_metrics

# ETF 宏观动量参数
ETF_POOL = ['QQQ', 'SPY', 'IWM', 'EFA', 'EEM']  # 进攻型
ETF_SAFE = ['GLD', 'TLT', 'SHY']                  # 防御型
ETF_MOM_LB = 126    # 6m 动量
ETF_TOP_N  = 2      # 选 Top-2
ETF_MA_WIN = 200    # 200dMA 趋势过滤
COST = 0.0015


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


def load_etf_data():
    """加载所有ETF数据"""
    etfs = {}
    for name in ETF_POOL + ETF_SAFE:
        fp = CACHE / f"{name}.csv"
        if not fp.exists():
            fp = STOCK_CACHE / f"{name}.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            c = 'Date' if 'Date' in df.columns else df.columns[0]
            df[c] = pd.to_datetime(df[c])
            df = df.set_index(c).sort_index()
            etfs[name] = pd.to_numeric(df['Close'], errors='coerce').dropna()
    return etfs


def run_etf_momentum_daily(etf_prices, start='2015-01-01', end='2025-12-31'):
    """
    ETF 宏观动量策略：
    月频信号 + 日频净值追踪
    
    规则：
      1. 计算 ETF_POOL 中各 ETF 的 6m 动量
      2. 只考虑 price > 200dMA 的 ETF
      3. 选 Top-2（如果没有正动量，全部 SHY）
      4. 等权分配 Top-2
    """
    # 构建价格矩阵
    all_etfs = list(set(ETF_POOL + ETF_SAFE))
    price_df = pd.DataFrame({name: etf_prices[name] for name in all_etfs if name in etf_prices})
    price_df = price_df.loc[start:end].dropna(how='all')
    
    # 计算 200dMA
    ma200 = price_df.rolling(ETF_MA_WIN, min_periods=int(ETF_MA_WIN*0.8)).mean()
    
    trading_days = price_df.index
    month_ends = price_df.resample('ME').last().index
    
    val = 1.0
    equity_vals, equity_dates = [], []
    current_weights = {}
    prev_w = {}
    processed = set()
    
    for day_idx, day in enumerate(trading_days):
        # 月末更新信号
        past_mes = month_ends[month_ends < day]
        if len(past_mes) > 0:
            last_me = past_mes[-1]
            if last_me not in processed:
                # 计算动量（使用月末前数据）
                mom_scores = {}
                for etf in ETF_POOL:
                    if etf not in price_df.columns: continue
                    hist = price_df[etf].loc[:last_me].dropna()
                    if len(hist) < ETF_MOM_LB + 5: continue
                    
                    # 6m 动量
                    mom = float(hist.iloc[-1] / hist.iloc[-ETF_MOM_LB] - 1)
                    
                    # 200dMA 过滤
                    ma_hist = ma200[etf].loc[:last_me].dropna()
                    if len(ma_hist) > 0 and float(hist.iloc[-1]) > float(ma_hist.iloc[-1]):
                        if mom > 0:  # 只要正动量
                            mom_scores[etf] = mom
                
                if len(mom_scores) >= ETF_TOP_N:
                    # 选 Top-N
                    sorted_etfs = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)
                    top_n = sorted_etfs[:ETF_TOP_N]
                    new_w = {etf: 1.0/ETF_TOP_N for etf, _ in top_n}
                elif len(mom_scores) > 0:
                    # 不足 Top-N → 全部选中 + 剩余 SHY
                    new_w = {etf: 1.0/ETF_TOP_N for etf, _ in mom_scores.items()}
                    shy_frac = 1.0 - sum(new_w.values())
                    if shy_frac > 0.01:
                        new_w['SHY'] = shy_frac
                else:
                    # 全部动量为负 → 安全港
                    # 检查 GLD 和 TLT 动量，选更好的
                    gld_hist = price_df['GLD'].loc[:last_me].dropna() if 'GLD' in price_df.columns else pd.Series(dtype=float)
                    tlt_hist = price_df['TLT'].loc[:last_me].dropna() if 'TLT' in price_df.columns else pd.Series(dtype=float)
                    
                    gld_mom = float(gld_hist.iloc[-1] / gld_hist.iloc[-ETF_MOM_LB] - 1) if len(gld_hist) >= ETF_MOM_LB + 5 else -1
                    tlt_mom = float(tlt_hist.iloc[-1] / tlt_hist.iloc[-ETF_MOM_LB] - 1) if len(tlt_hist) >= ETF_MOM_LB + 5 else -1
                    
                    if gld_mom > 0 and gld_mom >= tlt_mom:
                        new_w = {'GLD': 0.50, 'SHY': 0.50}
                    elif tlt_mom > 0:
                        new_w = {'TLT': 0.50, 'SHY': 0.50}
                    else:
                        new_w = {'SHY': 1.0}
                
                # 换仓成本
                all_t = set(new_w) | set(prev_w)
                turnover = sum(abs(new_w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
                val *= (1 - turnover * COST * 2)
                
                current_weights = new_w.copy()
                prev_w = new_w.copy()
                processed.add(last_me)
        
        # 日频净值
        if day_idx == 0:
            equity_vals.append(val); equity_dates.append(day)
            continue
        
        prev_day = trading_days[day_idx - 1]
        day_ret = 0.0
        
        for etf, w in current_weights.items():
            if etf in price_df.columns:
                if prev_day in price_df.index and day in price_df.index:
                    p0 = price_df[etf].loc[prev_day]
                    p1 = price_df[etf].loc[day]
                    if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                        day_ret += (p1/p0 - 1) * w
        
        val *= (1 + day_ret)
        equity_vals.append(val); equity_dates.append(day)
    
    return pd.Series(equity_vals, index=equity_dates, name='ETF_Momentum')


def build_3way(stock_eq, crypto_eq, etf_eq, w_stock, w_crypto, w_etf):
    """三组件混合"""
    idx = stock_eq.index.intersection(crypto_eq.index).intersection(etf_eq.index).sort_values()
    s_r = stock_eq.loc[idx].pct_change().fillna(0)
    c_r = crypto_eq.loc[idx].pct_change().fillna(0)
    e_r = etf_eq.loc[idx].pct_change().fillna(0)
    combo = (1 + w_stock*s_r + w_crypto*c_r + w_etf*e_r).cumprod()
    return combo


def main():
    print("=" * 78)
    print("🐻 Hybrid v8 — 三组件：v11b股票 + Crypto v5-D + ETF宏观动量")
    print("Direction D: 跨资产动量作为独立第三组件")
    print("=" * 78)
    
    # 加载数据
    print("\n[1/5] 加载数据...")
    v4_eq = pd.read_csv(BASE/'hybrid/codebear/hybrid_v4_equity.csv',
                         index_col=0, parse_dates=True)
    stock_eq = v4_eq['StockV11b'].dropna()
    
    btc_p = load_series('BTC_USD')
    eth_p = load_series('ETH_USD')
    gld_p = load_series('GLD')
    
    etf_prices = load_etf_data()
    print(f"  Stock v11b: {len(stock_eq)} 日")
    print(f"  ETFs available: {list(etf_prices.keys())}")
    
    # [2/5] ETF 独立回测
    print("\n[2/5] ETF 宏观动量策略独立回测...")
    etf_eq = run_etf_momentum_daily(etf_prices)
    em = calc_metrics(etf_eq)
    ewf = walk_forward(etf_eq)
    print(f"  ETF Momentum: CAGR={em['cagr']:.1%}, MaxDD={em['max_dd']:.1%}, "
          f"Sharpe={em['sharpe']:.2f}, WF={ewf['wf']:.3f}")
    print(f"    IS_Sh={ewf['is_sh']:.2f}→OOS_Sh={ewf['oos_sh']:.2f}  "
          f"IS_CAGR={ewf['is_cagr']:.1%}  OOS_CAGR={ewf['oos_cagr']:.1%}")
    
    # [3/5] Crypto v5-D
    print("\n[3/5] Crypto v5-D...")
    crypto_eq = run_crypto_v5_daily(btc_p, eth_p, gld_p, strategy='D')
    cm = calc_metrics(crypto_eq)
    cwf = walk_forward(crypto_eq)
    print(f"  Crypto v5-D: CAGR={cm['cagr']:.1%}, MaxDD={cm['max_dd']:.1%}, "
          f"Sharpe={cm['sharpe']:.2f}, WF={cwf['wf']:.3f}")
    
    # [4/5] 各组件指标
    print("\n[4/5] 各组件指标汇总...")
    sm = calc_metrics(stock_eq)
    swf = walk_forward(stock_eq)
    print(f"  Stock v11b:  CAGR={sm['cagr']:>7.1%}  MaxDD={sm['max_dd']:>7.1%}  "
          f"Sharpe={sm['sharpe']:>6.2f}  WF={swf['wf']:.3f}")
    print(f"  Crypto v5-D: CAGR={cm['cagr']:>7.1%}  MaxDD={cm['max_dd']:>7.1%}  "
          f"Sharpe={cm['sharpe']:>6.2f}  WF={cwf['wf']:.3f}")
    print(f"  ETF Momentum:CAGR={em['cagr']:>7.1%}  MaxDD={em['max_dd']:>7.1%}  "
          f"Sharpe={em['sharpe']:>6.2f}  WF={ewf['wf']:.3f}")
    
    # 相关性分析
    common = stock_eq.index.intersection(crypto_eq.index).intersection(etf_eq.index)
    s_r = stock_eq.loc[common].pct_change().dropna()
    c_r = crypto_eq.loc[common].pct_change().dropna()
    e_r = etf_eq.loc[common].pct_change().dropna()
    corr_sc = float(s_r.corr(c_r))
    corr_se = float(s_r.corr(e_r))
    corr_ce = float(c_r.corr(e_r))
    print(f"\n  日收益率相关性：")
    print(f"    Stock-Crypto: {corr_sc:.3f}")
    print(f"    Stock-ETF:    {corr_se:.3f}")
    print(f"    Crypto-ETF:   {corr_ce:.3f}")
    
    # [5/5] 三组件扫描
    print("\n[5/5] 三组件权重扫描...")
    
    # 2-way reference (v6 style)
    weight_configs = [
        # (w_stock, w_crypto, w_etf, label)
        (1.00, 0.00, 0.00, 'Pure Stock'),
        (0.85, 0.15, 0.00, 'v6 baseline (85:15:0)'),
        (0.80, 0.20, 0.00, '80:20:0'),
        # 3-way configs
        (0.70, 0.15, 0.15, '70:15:15'),
        (0.65, 0.15, 0.20, '65:15:20'),
        (0.60, 0.15, 0.25, '60:15:25'),
        (0.60, 0.20, 0.20, '60:20:20'),
        (0.55, 0.20, 0.25, '55:20:25'),
        (0.50, 0.25, 0.25, '50:25:25'),
        (0.50, 0.20, 0.30, '50:20:30'),
        (0.70, 0.10, 0.20, '70:10:20'),
        (0.65, 0.10, 0.25, '65:10:25'),
        (0.60, 0.10, 0.30, '60:10:30'),
        (0.75, 0.15, 0.10, '75:15:10'),
        (0.70, 0.20, 0.10, '70:20:10'),
        (0.65, 0.20, 0.15, '65:20:15'),
    ]
    
    print(f"\n{'配置':<22} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} "
          f"{'Comp':>8} {'WF':>7} {'IS_Sh':>7} {'OOS_Sh':>7}")
    print("-" * 100)
    
    best_comp = -99
    best_cfg = None
    results = []
    
    for ws, wc, we, label in weight_configs:
        if we == 0:
            if wc == 0:
                h_eq = stock_eq
            else:
                idx2 = stock_eq.index.intersection(crypto_eq.index).sort_values()
                s_r2 = stock_eq.loc[idx2].pct_change().fillna(0)
                c_r2 = crypto_eq.loc[idx2].pct_change().fillna(0)
                h_eq = (1 + ws*s_r2 + wc*c_r2).cumprod()
        else:
            h_eq = build_3way(stock_eq, crypto_eq, etf_eq, ws, wc, we)
        
        hm = calc_metrics(h_eq)
        hw = walk_forward(h_eq)
        comp = hm['sharpe']*0.4 + hm['calmar']*0.4 + min(hm['cagr'],1.0)*0.2
        
        wf_flag = "✅" if hw['wf'] >= 0.60 else ("⚠️" if hw['wf'] >= 0.55 else "❌")
        flag = ""
        if hw['wf'] >= 0.60 and comp > best_comp:
            best_comp = comp
            best_cfg = dict(label=label, ws=ws, wc=wc, we=we,
                           cagr=hm['cagr'], max_dd=hm['max_dd'],
                           sharpe=hm['sharpe'], calmar=hm['calmar'],
                           composite=comp, **hw)
            flag = " ★"
        
        print(f"  {label:<20} {hm['cagr']:>7.1%}  {hm['max_dd']:>7.1%}  "
              f"{hm['sharpe']:>7.2f}  {hm['calmar']:>7.2f}  {comp:>7.3f}  "
              f"{wf_flag}{hw['wf']:.2f}  {hw['is_sh']:>6.2f}  {hw['oos_sh']:>6.2f}{flag}")
        
        results.append(dict(label=label, ws=ws, wc=wc, we=we,
                            cagr=hm['cagr'], max_dd=hm['max_dd'],
                            sharpe=hm['sharpe'], calmar=hm['calmar'],
                            composite=comp, **hw))
    
    # 结论
    print("\n" + "=" * 100)
    v6_ref = dict(cagr=0.452, max_dd=-0.189, composite=1.382, wf=0.644)
    
    if best_cfg:
        print(f"\n🏆 v8 最优（WF≥0.60）：{best_cfg['label']}")
        print(f"   CAGR={best_cfg['cagr']:.1%} / MaxDD={best_cfg['max_dd']:.1%} / "
              f"WF={best_cfg['wf']:.3f} / Sharpe={best_cfg['sharpe']:.3f} / "
              f"Composite={best_cfg['composite']:.3f}")
        
        if best_cfg['cagr'] > 0.48 and best_cfg['wf'] > 0.62:
            print("\n🚀🚀🚀 【重大突破】CAGR>48% + WF>0.62！")
    else:
        print("\n⚠️  没有 WF≥0.60 的配置")
    
    # 保存
    out = {
        'components': {
            'stock_v11b': {**sm, **swf},
            'crypto_v5d': {**cm, **cwf},
            'etf_momentum': {**em, **ewf}
        },
        'correlations': {'stock_crypto': corr_sc, 'stock_etf': corr_se, 'crypto_etf': corr_ce},
        'sweep_results': results,
        'best': best_cfg,
        'meta': {'date': '2026-02-22', 'desc': 'Direction D: 3-way hybrid (stock + crypto + ETF momentum)'}
    }
    jf = Path(__file__).parent / "hybrid_v8_results.json"
    jf.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n💾 → {jf}")
    
    return out


if __name__ == '__main__':
    main()
