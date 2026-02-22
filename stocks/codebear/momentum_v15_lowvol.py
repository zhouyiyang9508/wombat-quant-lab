"""
Momentum v15 — 低波动因子 & 综合信号实验
==========================================
低波动异象（Low Volatility Anomaly）：
  - 经典研究：Baker et al. (2011) — 低波动股票 risk-adjusted 收益 > 高波动
  - 原因：投机需求 + 杠杆限制 + 委托代理问题
  - S&P 500 低波动指数（USMV）长期跑赢大盘

实验矩阵：
  Baseline  : v11b（纯动量排名）
  LV-A      : 纯低波动（用1年日收益波动率的倒数排名，选最低波动股票）
  LV-B      : 动量 + 低波动 50/50 混合评分
  LV-C      : 动量 × 低波动（乘积，同时满足）
  LV-D      : 动量筛选 + 低波动排名（先用动量过滤正值，再按低波动排序）

代码熊 🐻 | 2026-02-22
"""
import pandas as pd, numpy as np, json, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / 'data_cache'
STOCKS = CACHE / 'stocks'
sys.path.insert(0, str(BASE))
import stocks.codebear.momentum_v11b_final as v11b

def lc(name):
    fp = CACHE / f'{name}.csv'
    if not fp.exists(): return pd.Series(dtype=float)
    df = pd.read_csv(fp); c = 'Date' if 'Date' in df.columns else df.columns[0]
    df[c] = pd.to_datetime(df[c])
    return df.set_index(c).sort_index()['Close'].dropna()

def wf_score(eq, is_end='2021-12-31', oos_start='2022-01-01', rf=0.04):
    is_eq  = eq[eq.index <= pd.Timestamp(is_end)]
    oos_eq = eq[eq.index >= pd.Timestamp(oos_start)]
    if len(is_eq) < 12 or len(oos_eq) < 12: return 0, 0, 0
    def sh(e):
        r = e.pct_change().dropna()
        ann_r = (e.iloc[-1]/e.iloc[0])**(252/max(len(r),1)) - 1
        ann_v = r.std() * np.sqrt(252)
        return float((ann_r - rf) / ann_v) if ann_v > 0 else 0
    i, o = sh(is_eq), sh(oos_eq)
    return round(o/i, 3) if i > 0 else 0, round(i, 3), round(o, 3)

# ─── 自定义 select 函数（注入到 v11b）──────────────────────────────────────

def make_select_lv(mode, orig_select, daily_close_df):
    """
    包装 v11b.select，在股票选择阶段用不同的评分函数
    mode: 'lowvol' | 'combo50' | 'product' | 'filter_then_lv'
    """
    # 预计算：每只股票每个月的过去252日波动率（用月末前252日收益）
    # 用日频数据，更准确
    
    def select_fn(sig, sectors, date, prev_hold, gld_p, gdx_p, tlt_p, ief_p, def_prices):
        # 先让 v11b 做 regime 判断（保留债券/GLD/defensive 逻辑）
        # 但替换股票排名部分
        
        # 调用原始 select 获取 regime 信息
        orig_w, regime, bond_t = orig_select(sig, sectors, date, prev_hold,
                                              gld_p, gdx_p, tlt_p, ief_p, def_prices)
        
        # 如果 regime 不是 bull/soft_bull（比如 bear → 持债），直接返回
        # 判断：如果 bond_t 占比很高或股票权重很低，直接用原结果
        stock_total = sum(w for s, w in orig_w.items()
                         if s not in ('GLD','GDX','GDXJ','SHY','TLT','IEF','XLV','XLP','XLU'))
        if stock_total < 0.3:
            return orig_w, regime, bond_t
        
        # 重新计算股票评分
        tickers = [t for t in sig.index if t != 'SPY']
        
        # 获取动量分（原始）
        mom_scores = {}
        for t in tickers:
            if t in sig.index:
                s_row = sig.loc[t]
                if 'mom' in s_row.index and not pd.isna(s_row['mom']):
                    mom_scores[t] = s_row['mom']
        
        # 获取低波动分（日频数据）
        lv_scores = {}
        if daily_close_df is not None:
            mask = daily_close_df.index <= date
            hist = daily_close_df[mask].tail(252)
            for t in tickers:
                if t in hist.columns:
                    r = hist[t].pct_change().dropna()
                    if len(r) > 60:
                        lv_scores[t] = 1.0 / (r.std() * np.sqrt(252) + 1e-6)  # 逆年化波动率
        
        if not lv_scores or not mom_scores:
            return orig_w, regime, bond_t
        
        # 标准化到 [0,1]
        def normalize(d):
            vals = list(d.values())
            mn, mx = min(vals), max(vals)
            if mx == mn: return {k: 0.5 for k in d}
            return {k: (v-mn)/(mx-mn) for k, v in d.items()}
        
        mom_n = normalize({t: v for t, v in mom_scores.items() if t in lv_scores})
        lv_n  = normalize({t: v for t, v in lv_scores.items() if t in mom_n})
        
        common = set(mom_n.keys()) & set(lv_n.keys())
        if not common:
            return orig_w, regime, bond_t
        
        # 合成评分
        if mode == 'lowvol':
            scores = {t: lv_n[t] for t in common if mom_scores.get(t, 0) > -999}
        elif mode == 'combo50':
            scores = {t: 0.5*mom_n[t] + 0.5*lv_n[t] for t in common}
        elif mode == 'product':
            scores = {t: mom_n[t] * lv_n[t] for t in common}
        elif mode == 'filter_then_lv':
            # 只选动量为正的股票，然后按低波动排名
            pos = {t: lv_n[t] for t in common if mom_scores.get(t, 0) > 0}
            scores = pos if pos else {t: lv_n[t] for t in common}
        else:
            return orig_w, regime, bond_t
        
        # 重新按 sector 分组，选 top
        df = pd.DataFrame({'score': scores})
        df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
        df = df[df['sector'] != 'Unknown']
        df = df[df['score'] > 0]
        
        if df.empty:
            return orig_w, regime, bond_t
        
        # 保留 regime 中的行业逻辑（用原始 weight 判断选了哪些行业）
        orig_stocks = {s: w for s, w in orig_w.items()
                      if s not in ('GLD','GDX','GDXJ','SHY','TLT','IEF','XLV','XLP','XLU')}
        
        # 复制原始的股票持仓结构，但用新评分替换权重
        # 保持相同的行业分布（从 orig_w 读取行业结构）
        orig_sectors_used = set()
        for s in orig_stocks:
            sec = sectors.get(s, 'Unknown')
            if sec != 'Unknown':
                orig_sectors_used.add(sec)
        
        if not orig_sectors_used:
            return orig_w, regime, bond_t
        
        # 在原始选中的行业内，按新评分重新选股
        new_stocks = {}
        for sec in orig_sectors_used:
            sec_df = df[df['sector'] == sec].sort_values('score', ascending=False)
            n_select = max(1, sum(1 for s in orig_stocks if sectors.get(s) == sec))
            for t in sec_df.head(n_select).index:
                new_stocks[t] = sec_df.loc[t, 'score']
        
        if not new_stocks:
            return orig_w, regime, bond_t
        
        # 逆波动率加权（保持与 v11b 相同的加权方式）
        inv_vol = {t: lv_scores[t] for t in new_stocks if t in lv_scores}
        if not inv_vol:
            return orig_w, regime, bond_t
        total_iv = sum(inv_vol.values())
        
        # 非股票权重（债券/GLD等）保持不变
        non_stock_w = {s: w for s, w in orig_w.items() if s not in orig_stocks}
        stock_budget = stock_total
        
        new_w = dict(non_stock_w)
        for t, iv in inv_vol.items():
            new_w[t] = stock_budget * (iv / total_iv)
        
        return new_w, regime, bond_t
    
    return select_fn


if __name__ == '__main__':
    print("=" * 68)
    print("🐻 v15 — 低波动因子 vs 纯动量对比")
    print("=" * 68)
    
    # 加载数据
    print("\n[1/3] 加载数据...")
    sp500_tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    sp500_sectors = json.load(open(CACHE / "sp500_sectors.json"))
    
    gld_p  = lc('GLD'); gdx_p = lc('GDX'); gdxj_p = lc('GDXJ')
    shy_p  = lc('SHY'); tlt_p = lc('TLT'); ief_p  = lc('IEF')
    xlv_p  = lc('XLV'); xlp_p = lc('XLP'); xlu_p  = lc('XLU')
    def_prices = {'XLV': xlv_p, 'XLP': xlp_p, 'XLU': xlu_p}
    
    close_df = v11b.load_stocks(sp500_tickers + ['SPY'])
    close_df = close_df.loc['2015-01-01':'2025-12-31']
    
    # 预计算
    print("[2/3] 预计算动量信号...")
    sig = v11b.precompute(close_df)
    
    print("[3/3] 运行实验矩阵...\n")
    
    orig_select = v11b.select  # 保存原始 select

    configs = [
        ('baseline',      '📊 Baseline v11b（纯动量）',                None),
        ('combo50',       '📊 LV-B 动量+低波动 50/50',               'combo50'),
        ('filter_then_lv','📊 LV-D 动量筛选→低波动排名',             'filter_then_lv'),
        ('lowvol',        '📊 LV-A 纯低波动',                        'lowvol'),
        ('product',       '📊 LV-C 动量×低波动（乘积）',             'product'),
    ]
    
    results = []
    for key, label, mode in configs:
        if mode is None:
            # Baseline：直接运行
            v11b.select = orig_select
        else:
            new_sel = make_select_lv(mode, orig_select, close_df)
            v11b.select = new_sel
        
        try:
            eq, avg_to, rh, bh = v11b.run_backtest(
                close_df, sig, sp500_sectors,
                gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices)
            m = v11b.compute_metrics(eq)
            wfr, is_sh, oos_sh = wf_score(eq)
            comp = m['sharpe']*0.4 + m['calmar']*0.4 + min(m['cagr'],1.0)*0.2
            flag = '✅' if wfr >= 0.60 else '⚠️' if wfr >= 0.55 else '❌'
            
            print(f"{label}")
            print(f"  CAGR={m['cagr']*100:.1f}%  MaxDD={m['max_dd']*100:.1f}%  "
                  f"Sh={m['sharpe']:.2f}  Cal={m['calmar']:.2f}  Comp={comp:.3f}  "
                  f"WF={wfr}{flag}  IS={is_sh:.2f}→OOS={oos_sh:.2f}  TO={avg_to*100:.1f}%\n")
            results.append((label, m, wfr, comp))
        except Exception as e:
            print(f"{label}: ERROR — {e}\n")
        finally:
            v11b.select = orig_select  # 恢复
    
    print("=" * 68)
    if results:
        best = max(results, key=lambda x: x[-1])
        print(f"🏆 最优: {best[0]}")
        print(f"   Comp={best[-1]:.3f}")
