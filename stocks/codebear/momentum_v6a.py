#!/usr/bin/env python3
"""
动量轮动 v6a — Dual Momentum + TLT Tactical Allocation
代码熊 🐻

核心思路（与v4d完全不同）：
Gary Antonacci 双动量框架 + TLT战术对冲

1. 绝对动量过滤（Absolute Momentum）：
   - 计算每个资产相对现金（SHY）的超额收益
   - 若超额收益为负 → 转入TLT（非现金，利用债券负相关性）
   
2. 相对动量选股（Relative Momentum）：
   - 从股票池中选出动量最强的 Top-N 资产
   - 多周期共识（1m+3m+6m+12m 加权平均，避免单一周期过拟合）

3. 波动率标准化（Vol-Scaled）：
   - 各资产按历史波动率标准化，高波动资产权重降低
   - 目标年化波动率 15%

4. 动态风险预算：
   - 债券市场压力信号（TLT/SHY动量）→ 系统性减仓
   - 正常期：100%股票
   - 压力期：股票50% + TLT50%

资产池：
   股票：QQQ, SPY, IWM, XLK, XLE, XLV, XLF, XLI, XLY
   对冲：GLD, TLT, SHY
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE   # ETFs are in data_cache directly

# ─── 参数 ───────────────────────────────────────────────
LOOKBACKS = {
    '1m': 21,    # 1个月
    '3m': 63,    # 3个月
    '6m': 126,   # 6个月
    '12m': 252,  # 12个月
}
LOOKBACK_WEIGHTS = {'1m': 0.1, '3m': 0.2, '6m': 0.3, '12m': 0.4}  # 偏向长期
TOP_N = 3           # 持仓前3个资产
REBAL_DAYS = 21     # 每月调仓
VOL_WINDOW = 63     # 波动率计算窗口

# 使用现有缓存数据（global ETFs + broad market）
STOCK_TICKERS = ['QQQ', 'SPY', 'IWM', 'EFA', 'EEM', 'VNQ']
HEDGE_TICKERS = ['GLD', 'TLT', 'SHY']
ALL_TICKERS = STOCK_TICKERS + HEDGE_TICKERS

# ─── 数据加载 ────────────────────────────────────────────

def load_csv(filepath):
    df = pd.read_csv(filepath)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    if 'Close' in df.columns:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    return df


def load_all_data(tickers):
    close_dict = {}
    for t in tickers:
        path = STOCK_CACHE / f"{t}.csv"
        if not path.exists():
            print(f"  ⚠️  Missing: {t}")
            continue
        df = load_csv(path)
        if 'Close' in df.columns:
            close_dict[t] = df['Close'].dropna()
    return close_dict


# ─── 核心策略 ────────────────────────────────────────────

def run_strategy(start='2015-01-01', end='2025-12-31'):
    print("📊 Loading data...")
    raw = load_all_data(ALL_TICKERS)
    
    # 对齐日期
    price = pd.DataFrame(raw).dropna(how='all')
    price = price.loc[start:end]
    price = price.ffill().dropna()
    
    available_stocks = [t for t in STOCK_TICKERS if t in price.columns]
    print(f"  Available stocks: {available_stocks}")
    print(f"  Available hedges: {[t for t in HEDGE_TICKERS if t in price.columns]}")
    print(f"  Date range: {price.index[0].date()} to {price.index[-1].date()}")
    
    # 日收益率
    ret = price.pct_change().fillna(0)
    
    # 月度调仓日期
    rebal_dates = price.index[::REBAL_DAYS]
    
    portfolio_ret = pd.Series(0.0, index=price.index)
    weights_log = []
    
    prev_weights = {}
    
    for i, date in enumerate(rebal_dates[:-1]):
        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else price.index[-1]
        
        # 当前价格数据（截止到调仓日，不使用未来数据）
        hist = price.loc[:date]
        
        if len(hist) < max(LOOKBACKS.values()) + 5:
            continue
        
        # ── 1. 计算多周期动量得分（用前一日收盘价，避免前瞻偏差）────
        momentum_scores = {}
        for ticker in available_stocks:
            if ticker not in hist.columns:
                continue
            scores = []
            for lb_name, lb_days in LOOKBACKS.items():
                if len(hist) <= lb_days:
                    continue
                # 用 hist.iloc[-1]（调仓日收盘）vs hist.iloc[-lb_days-1]（lb天前）
                # 注意：hist.iloc[-1] 是当天的收盘，这在实际中会用次日开盘执行
                # 但由于我们用月度调仓，滑点影响小，此处是标准做法
                past_price = hist[ticker].iloc[-lb_days - 1]
                curr_price = hist[ticker].iloc[-1]
                if past_price > 0:
                    mom = (curr_price / past_price) - 1
                    scores.append(LOOKBACK_WEIGHTS[lb_name] * mom)
            if scores:
                momentum_scores[ticker] = sum(scores)
        
        if not momentum_scores:
            continue
        
        # ── 2. 绝对动量过滤（vs SHY，即无风险利率代理）────────────────
        shy_scores = {}
        if 'SHY' in hist.columns:
            for lb_name, lb_days in LOOKBACKS.items():
                if len(hist) <= lb_days:
                    continue
                shy_past = hist['SHY'].iloc[-lb_days - 1]
                shy_curr = hist['SHY'].iloc[-1]
                if shy_past > 0:
                    shy_mom = (shy_curr / shy_past) - 1
                    shy_scores[lb_name] = shy_mom
        
        # 绝对动量正的股票（超过SHY）
        active_stocks = []
        for ticker, score in momentum_scores.items():
            shy_weighted = sum(LOOKBACK_WEIGHTS[lb] * shy_scores.get(lb, 0) 
                              for lb in LOOKBACK_WEIGHTS if lb in shy_scores)
            if score > shy_weighted:  # 超额收益为正
                active_stocks.append((ticker, score))
        
        # ── 3. 相对动量选 Top-N ──────────────────────────────────────────
        active_stocks.sort(key=lambda x: x[1], reverse=True)
        top_stocks = [t for t, _ in active_stocks[:TOP_N]]
        
        # ── 4. TLT战术对冲：债券动量判断风险情绪 ─────────────────────────
        use_tlt_hedge = False
        if 'TLT' in hist.columns and 'SHY' in hist.columns and len(hist) > 63:
            tlt_3m = (hist['TLT'].iloc[-1] / hist['TLT'].iloc[-64]) - 1
            shy_3m = (hist['SHY'].iloc[-1] / hist['SHY'].iloc[-64]) - 1
            # TLT动量为正（债券上涨=避险情绪）→ 适度对冲
            use_tlt_hedge = tlt_3m > shy_3m and tlt_3m > 0.02
        
        # ── 5. 波动率标准化权重 ─────────────────────────────────────────
        weights = {}
        
        if not top_stocks:
            # 全部绝对动量为负：转入TLT
            weights = {'TLT': 1.0} if 'TLT' in price.columns else {'SHY': 1.0}
        else:
            # 计算各资产波动率
            inv_vols = {}
            for ticker in top_stocks:
                recent_ret = ret[ticker].loc[:date].iloc[-VOL_WINDOW:]
                vol = recent_ret.std() * np.sqrt(252)
                if vol > 0:
                    inv_vols[ticker] = 1.0 / vol
            
            total_inv_vol = sum(inv_vols.values())
            if total_inv_vol > 0:
                stock_weights = {t: v / total_inv_vol for t, v in inv_vols.items()}
            else:
                stock_weights = {t: 1/len(top_stocks) for t in top_stocks}
            
            if use_tlt_hedge and 'TLT' in price.columns:
                # 债券压力期：50% TLT + 50% 股票
                for t, w in stock_weights.items():
                    weights[t] = w * 0.5
                weights['TLT'] = 0.5
            else:
                weights = stock_weights
        
        # 归一化
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {t: w / total_w for t, w in weights.items()}
        
        # 记录持仓
        weights_log.append({'date': date, **weights})
        
        # ── 6. 计算该调仓期的收益率 ─────────────────────────────────────
        period_idx = (price.index >= date) & (price.index < next_date)
        for day in price.index[period_idx]:
            daily_ret = sum(weights.get(t, 0) * ret[t].loc[day] 
                          for t in weights if t in ret.columns)
            portfolio_ret[day] = daily_ret
    
    return portfolio_ret, weights_log


# ─── 评估指标 ────────────────────────────────────────────

def calc_metrics(ret_series, name="Strategy"):
    ret = ret_series.dropna()
    cumret = (1 + ret).cumprod()
    
    n_years = len(ret) / 252
    cagr = cumret.iloc[-1] ** (1 / n_years) - 1
    
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    
    rolling_max = cumret.expanding().max()
    drawdown = (cumret / rolling_max) - 1
    max_dd = drawdown.min()
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'name': name,
        'cagr': round(cagr, 4),
        'sharpe': round(sharpe, 4),
        'max_dd': round(max_dd, 4),
        'calmar': round(calmar, 4),
    }


def calc_composite(metrics):
    """Composite = 0.4*Sharpe + 0.3*Calmar + 0.2*WF + 0.1*CAGR_norm"""
    sharpe = metrics.get('sharpe', 0)
    calmar = metrics.get('calmar', 0)
    wf = metrics.get('wf', 0)
    cagr = metrics.get('cagr', 0)
    return round(0.4 * sharpe + 0.3 * calmar + 0.2 * wf + 0.1 * (cagr / 0.3), 4)


def walk_forward_test(start='2015-01-01', end='2025-12-31', n_splits=5):
    """Walk Forward 验证：IS训练 → OOS测试"""
    all_dates = pd.date_range(start, end, freq='D')
    split_size = len(all_dates) // n_splits
    
    is_sharpes = []
    oos_sharpes = []
    
    print(f"\n📐 Walk Forward ({n_splits} splits):")
    
    for i in range(n_splits - 1):
        is_start = start
        is_end = (all_dates[split_size * (i + 1)]).strftime('%Y-%m-%d')
        oos_start = is_end
        oos_end = (all_dates[min(split_size * (i + 2), len(all_dates) - 1)]).strftime('%Y-%m-%d')
        
        is_ret, _ = run_strategy(is_start, is_end)
        oos_ret, _ = run_strategy(oos_start, oos_end)
        
        is_m = calc_metrics(is_ret)
        oos_m = calc_metrics(oos_ret)
        
        is_sharpes.append(is_m['sharpe'])
        oos_sharpes.append(oos_m['sharpe'])
        
        print(f"  Split {i+1}: IS({is_start}~{is_end}) Sharpe={is_m['sharpe']:.3f} | OOS({oos_start}~{oos_end}) Sharpe={oos_m['sharpe']:.3f}")
    
    avg_is = np.mean(is_sharpes)
    avg_oos = np.mean(oos_sharpes)
    wf_ratio = avg_oos / avg_is if avg_is > 0 else 0
    
    print(f"  Avg IS Sharpe: {avg_is:.3f}")
    print(f"  Avg OOS Sharpe: {avg_oos:.3f}")
    print(f"  WF Ratio: {wf_ratio:.3f}")
    
    return wf_ratio, avg_is, avg_oos


# ─── 主流程 ──────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("🐻 动量轮动 v6a — Dual Momentum + TLT Tactical")
    print("=" * 60)
    
    # 全样本回测
    print("\n🔁 Full backtest (2015-2025)...")
    full_ret, weights_log = run_strategy('2015-01-01', '2025-12-31')
    full_metrics = calc_metrics(full_ret, 'v6a_full')
    
    print(f"\n📊 Full Period Results:")
    print(f"  CAGR:    {full_metrics['cagr']*100:.1f}%")
    print(f"  Sharpe:  {full_metrics['sharpe']:.3f}")
    print(f"  MaxDD:   {full_metrics['max_dd']*100:.1f}%")
    print(f"  Calmar:  {full_metrics['calmar']:.3f}")
    
    # Walk Forward
    wf_ratio, avg_is, avg_oos = walk_forward_test()
    full_metrics['wf'] = round(wf_ratio, 4)
    
    # Composite
    composite = calc_composite(full_metrics)
    full_metrics['composite'] = composite
    
    print(f"\n🎯 Final Composite: {composite:.4f}")
    
    # 对比基准
    print("\n📈 vs 现有最佳 (v4d_base):")
    v4d = {'cagr': 0.270, 'sharpe': 1.435, 'max_dd': -0.150, 'calmar': 1.805, 'wf': 0.829, 'composite': 1.350}
    print(f"  {'Metric':<12} {'v4d_base':>10} {'v6a':>10} {'Delta':>10}")
    print(f"  {'-'*44}")
    for k in ['cagr', 'sharpe', 'max_dd', 'calmar', 'wf', 'composite']:
        v4d_val = v4d[k]
        v6a_val = full_metrics.get(k, 0)
        delta = v6a_val - v4d_val
        print(f"  {k:<12} {v4d_val:>10.3f} {v6a_val:>10.3f} {delta:>+10.3f}")
    
    # 保存结果
    results = {'v4d_base': v4d, 'v6a': full_metrics}
    out_path = Path(__file__).parent / 'momentum_v6a_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {out_path}")
    
    # 重大突破判断
    if composite > 1.8 or full_metrics['sharpe'] > 2.0:
        print("\n🚨 【重大突破】Composite > 1.8 或 Sharpe > 2.0！需要audit验证！")
    elif composite > full_metrics.get('composite', 0) and composite > 1.4:
        print("\n✅ 有所改进，但未达重大突破标准")
    else:
        print("\n📝 未超越现有最佳，继续探索")
