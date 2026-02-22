"""
ETF Trend Following — 全球资产类别趋势跟踪
===========================================
策略逻辑：
  - 宇宙：~20 只全球 ETF，覆盖股票/债券/商品/房地产/国际市场
  - 信号：每月末用 12-1 月动量排名（与 v11b 相同公式）
  - 持仓：做多前 N 只（动量为正的），按逆波动率加权
  - 防御：若正动量 ETF < N，用 SHY 补足；若全部负动量，全仓 SHY
  - 换手：月频
  - 对标：实盘 CTA / 管理期货

关键问题：
  1. 独立 CAGR/Sharpe 是多少？
  2. 与 v11b 股票策略相关性有多低？
  3. 混合后能否超越 Hybrid v6（CAGR=45.2%，WF=0.644）？

代码熊 🐻 | 2026-02-22
"""
import pandas as pd, numpy as np, json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / 'data_cache'

# ─── ETF 宇宙 ──────────────────────────────────────────────────────────────
ETF_UNIVERSE = {
    # 美国股票
    'SPY':  'US_Equity',    # S&P 500
    'QQQ':  'US_Equity',    # Nasdaq 100
    'IWM':  'US_Equity',    # Russell 2000 小盘
    # 国际股票
    'EFA':  'Intl_Equity',  # 发达市场（除美国）
    'EEM':  'EM_Equity',    # 新兴市场
    'VGK':  'Intl_Equity',  # 欧洲
    'EWJ':  'Intl_Equity',  # 日本
    'FXI':  'EM_Equity',    # 中国
    # 固定收益
    'TLT':  'LT_Bond',      # 长期美债
    'IEF':  'MT_Bond',      # 中期美债
    'HYG':  'HY_Bond',      # 高收益债
    'LQD':  'IG_Bond',      # 投资级债
    'TIP':  'TIPS',         # 通胀保护债
    # 商品
    'GLD':  'Commodity',    # 黄金
    'SLV':  'Commodity',    # 白银
    'DBC':  'Commodity',    # 商品综合
    'USO':  'Commodity',    # 原油
    # 房地产
    'VNQ':  'REIT',         # 美国 REIT
    # 矿业/特殊
    'GDX':  'Gold_Mining',  # 黄金矿业
    'GDXJ': 'Gold_Mining',  # 初级黄金矿业
}

TOP_N        = 6      # 持有前 N 只
MOM_SKIP     = 1      # 跳过最近1个月（避免短期反转）
MOM_WINDOW   = 12     # 动量窗口（月）
VOL_WINDOW   = 6      # 波动率窗口（月，用于逆波动率加权）
RF_ANNUAL    = 0.04   # 无风险利率
TC           = 0.0010 # 单边交易成本（ETF 更低，0.10%）

def load_etf(sym):
    fp = CACHE / f'{sym}.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    c = 'Date' if 'Date' in df.columns else df.columns[0]
    df[c] = pd.to_datetime(df[c])
    df = df.set_index(c).sort_index()
    if 'Close' not in df.columns:
        return None
    return df['Close'].dropna()

def compute_metrics(eq, rf=RF_ANNUAL):
    r = eq.pct_change().dropna()
    freq = 252  # daily
    ann_r = (eq.iloc[-1] / eq.iloc[0]) ** (freq / len(r)) - 1
    ann_v = r.std() * np.sqrt(freq)
    sharpe = (ann_r - rf) / ann_v if ann_v > 0 else 0
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max
    max_dd = dd.min()
    calmar = ann_r / abs(max_dd) if max_dd < 0 else 0
    return dict(cagr=ann_r, max_dd=max_dd, sharpe=sharpe, calmar=calmar)

def walk_forward(eq, is_end='2021-12-31', oos_start='2022-01-01'):
    is_eq  = eq[eq.index <= pd.Timestamp(is_end)]
    oos_eq = eq[eq.index >= pd.Timestamp(oos_start)]
    def sh(e):
        r = e.pct_change().dropna()
        ann_r = (e.iloc[-1]/e.iloc[0])**(252/len(r)) - 1
        ann_v = r.std() * np.sqrt(252)
        return float((ann_r - RF_ANNUAL) / ann_v) if ann_v > 0 else 0
    is_sh, oos_sh = sh(is_eq), sh(oos_eq)
    wf = round(oos_sh / is_sh, 3) if is_sh > 0 else 0
    return wf, round(is_sh, 3), round(oos_sh, 3)

def run_etf_trend(top_n=TOP_N, mom_window=MOM_WINDOW, vol_window=VOL_WINDOW):
    """运行 ETF 趋势跟踪策略，返回日频 equity 曲线"""
    # 加载数据
    prices = {}
    for sym in ETF_UNIVERSE:
        s = load_etf(sym)
        if s is not None and len(s) > 300:
            prices[sym] = s
    shy = load_etf('SHY')
    
    price_df = pd.DataFrame(prices).sort_index()
    price_df = price_df.loc['2014-12-01':]
    
    # 月频：取每月最后一个交易日
    monthly = price_df.resample('ME').last()
    
    available_syms = list(prices.keys())
    print(f"  可用 ETF: {len(available_syms)} 只 → {available_syms}")
    
    # 回测
    equity = [1.0]
    dates  = [monthly.index[mom_window]]
    prev_hold = {}  # sym → weight
    
    for i in range(mom_window, len(monthly)):
        dt = monthly.index[i]
        
        # 动量信号：(p[i-skip] / p[i-window]) - 1
        # skip=1 → p[i-1] / p[i-window-1+1] → p[i-1] / p[i-MOM_WINDOW]
        skip_i  = i - MOM_SKIP          # i-1
        start_i = i - mom_window        # i-12（含skip）
        
        moms = {}
        for sym in available_syms:
            col = monthly[sym]
            if col.isna().any() or col.iloc[start_i] <= 0:
                continue
            moms[sym] = col.iloc[skip_i] / col.iloc[start_i] - 1
        
        # 逆波动率权重（用过去 vol_window 个月的月收益）
        vols = {}
        for sym in moms:
            ret_slice = monthly[sym].pct_change().iloc[i-vol_window:i].dropna()
            if len(ret_slice) < 3: continue
            vols[sym] = ret_slice.std()
        
        # 选正动量中排名前 top_n
        pos_moms = {s: m for s, m in moms.items() if m > 0 and s in vols and vols[s] > 0}
        ranked   = sorted(pos_moms.items(), key=lambda x: -x[1])[:top_n]
        selected = [s for s, _ in ranked]
        
        # 权重：逆波动率
        if selected:
            inv_vol = {s: 1.0/vols[s] for s in selected}
            total_iv = sum(inv_vol.values())
            stock_weight = sum(inv_vol.values()) / total_iv  # 1.0（全仓 selected）
            
            # 不足 top_n 的部分用 SHY 填
            weights = {s: inv_vol[s]/total_iv for s in selected}
        else:
            weights = {}
        
        # 当前持仓 → 新持仓，计算下一个月的 P&L（日频追踪）
        if i < len(monthly) - 1:
            next_month_end = monthly.index[i+1]
            
            # 日频区间
            mask = (price_df.index > dt) & (price_df.index <= next_month_end)
            daily_slice = price_df[mask]
            shy_slice   = shy[shy.index.isin(daily_slice.index)] if shy is not None else None
            
            if len(daily_slice) == 0:
                equity.append(equity[-1])
                dates.append(next_month_end)
                prev_hold = weights
                continue
            
            # 交易成本（换仓部分）
            all_syms = set(list(weights.keys()) + list(prev_hold.keys()))
            tc_cost = sum(abs(weights.get(s, 0) - prev_hold.get(s, 0)) for s in all_syms) * TC
            
            # 每日收益
            ret = equity[-1] * (1 - tc_cost)
            for day_idx in range(len(daily_slice)):
                day_ret = 0.0
                for sym, w in weights.items():
                    if sym in daily_slice.columns:
                        col = daily_slice[sym]
                        if day_idx == 0:
                            p0 = price_df[sym][price_df[sym].index <= dt].iloc[-1]
                        else:
                            p0 = daily_slice[sym].iloc[day_idx - 1]
                        p1 = col.iloc[day_idx]
                        if p0 > 0:
                            day_ret += w * (p1/p0 - 1)
                
                # SHY 补足（1 - sum(weights)）
                shy_w = max(0, 1 - sum(weights.values()))
                if shy_slice is not None and len(shy_slice) > day_idx:
                    sp0 = shy[shy.index <= dt].iloc[-1] if day_idx == 0 else shy_slice.iloc[day_idx-1]
                    sp1 = shy_slice.iloc[day_idx]
                    if sp0 > 0:
                        day_ret += shy_w * (sp1/sp0 - 1)
                
                ret *= (1 + day_ret)
            
            equity.append(ret)
            dates.append(next_month_end)
        
        prev_hold = weights
    
    eq_series = pd.Series(equity, index=dates)
    eq_series = eq_series.loc['2015-01-01':'2025-12-31']
    return eq_series


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(BASE))
    
    print("=" * 65)
    print("🌍 ETF 趋势跟踪策略 — 全资产类别")
    print("=" * 65)
    
    # ── 参数扫描 ──────────────────────────────────────────────────────────
    configs = [
        (6, 12, 6,  "TopN=6, Mom=12m"),
        (4, 12, 6,  "TopN=4, Mom=12m"),
        (8, 12, 6,  "TopN=8, Mom=12m"),
        (6,  6, 3,  "TopN=6, Mom=6m"),
        (5,  9, 6,  "TopN=5, Mom=9m"),
    ]
    
    results = []
    best_eq = None
    
    for top_n, mom_w, vol_w, label in configs:
        print(f"\n  [{label}]...", flush=True)
        eq = run_etf_trend(top_n=top_n, mom_window=mom_w, vol_window=vol_w)
        m  = compute_metrics(eq)
        wfr, is_sh, oos_sh = walk_forward(eq)
        comp = m['sharpe']*0.4 + m['calmar']*0.4 + min(m['cagr'],1.0)*0.2
        flag = '✅' if wfr >= 0.60 else '⚠️' if wfr >= 0.50 else '❌'
        
        print(f"  → CAGR={m['cagr']*100:.1f}%  MaxDD={m['max_dd']*100:.1f}%  "
              f"Sh={m['sharpe']:.2f}  Cal={m['calmar']:.2f}  "
              f"Comp={comp:.3f}  WF={wfr}{flag}  IS={is_sh:.2f}→OOS={oos_sh:.2f}")
        results.append((label, m, wfr, is_sh, oos_sh, comp, eq))
        if best_eq is None or comp > results[-2][-2] if len(results) > 1 else True:
            best_eq = eq
    
    # 找最优
    best = max(results, key=lambda x: x[-2])
    best_label, best_m, best_wf, _, _, best_comp, best_eq = best
    
    print("\n" + "=" * 65)
    print(f"🏆 最优配置: {best_label}")
    print(f"   CAGR={best_m['cagr']*100:.1f}%  MaxDD={best_m['max_dd']*100:.1f}%  "
          f"Sh={best_m['sharpe']:.2f}  Cal={best_m['calmar']:.2f}  "
          f"Comp={best_comp:.3f}  WF={best_wf}")
    
    # 与 v11b 股票相关性
    v11b_eq_path = BASE / 'hybrid/codebear/hybrid_v4_equity.csv'
    if v11b_eq_path.exists():
        v11b_df = pd.read_csv(v11b_eq_path, index_col=0, parse_dates=True)
        v11b_eq = v11b_df['StockV11b'].dropna()
        
        # 对齐日期
        common = best_eq.index.intersection(v11b_eq.index)
        if len(common) > 100:
            r_etf  = best_eq.loc[common].pct_change().dropna()
            r_stk  = v11b_eq.loc[common].pct_change().dropna()
            common2 = r_etf.index.intersection(r_stk.index)
            corr = r_etf.loc[common2].corr(r_stk.loc[common2])
            print(f"\n📐 与 v11b 股票日收益相关性: {corr:.3f}")
            print(f"   (越低越好，理想值 < 0.3)")
    
    # 保存最优 equity
    out_path = BASE / 'strategies/codebear/etf_trend_equity.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'ETF_Trend': best_eq}).to_csv(out_path)
    print(f"\n✓ Equity 已保存 → {out_path.name}")
    
    # Hybrid 潜力预估（与 v11b 等权混合）
    if v11b_eq_path.exists():
        v11b_df = pd.read_csv(v11b_eq_path, index_col=0, parse_dates=True)
        v11b_eq = v11b_df['StockV11b'].dropna()
        common = best_eq.index.intersection(v11b_eq.index)
        if len(common) > 100:
            e1 = best_eq.loc[common] / best_eq.loc[common].iloc[0]
            e2 = v11b_eq.loc[common] / v11b_eq.loc[common].iloc[0]
            for w_etf in [0.20, 0.30, 0.40, 0.50]:
                mix = w_etf * e1 + (1-w_etf) * e2
                mm  = compute_metrics(mix)
                wfr_mix, is_mix, oos_mix = walk_forward(mix)
                flag = '✅' if wfr_mix >= 0.60 else '⚠️'
                print(f"  混合 w_ETF={int(w_etf*100)}%: CAGR={mm['cagr']*100:.1f}%  "
                      f"MaxDD={mm['max_dd']*100:.1f}%  Sh={mm['sharpe']:.2f}  "
                      f"WF={wfr_mix}{flag}")
