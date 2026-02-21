#!/usr/bin/env python3
"""
动量轮动 v6b — 多因子 + TLT/GLD 双对冲
代码熊 🐻

核心差异（vs v4d）：
1. 选股因子：动量 + 低波动率复合因子（不只是动量）
   - 高动量 + 低波动率的股票往往有更高的风险调整收益
   - 避免选入"高动量但极高波动"的投机股
   
2. 双对冲机制（TLT + GLD）：
   - 根据宏观环境选择最优避险资产
   - 通胀上升期（GLD动量 > TLT动量）：用GLD对冲
   - 经济下行期（TLT动量 > GLD动量）：用TLT对冲
   - 正常期：100% 个股
   
3. 回撤阈值更保守（-6%开始对冲，更早介入）

资产池：S&P 500个股（同v4d）+ GLD + TLT
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# ─── 参数 ───────────────────────────────────────────────
MOM_WINDOW   = 126      # 6个月动量
VOL_WINDOW   = 63       # 3个月波动率
SMA_WINDOW   = 50       # SMA50趋势过滤
REBAL_DAYS   = 21       # 月度调仓

TOP_N        = 15       # 持仓股票数量
MOM_WEIGHT   = 0.6      # 动量因子权重
VOL_WEIGHT   = 0.4      # 低波动因子权重

# 回撤响应式双对冲
DD_THRESHOLD_1 = -0.06  # -6%开始对冲（比v4d的-8%更早）
DD_THRESHOLD_2 = -0.10
DD_THRESHOLD_3 = -0.16

HEDGE_ALLOC_1 = 0.25    # 一级对冲
HEDGE_ALLOC_2 = 0.45    # 二级对冲
HEDGE_ALLOC_3 = 0.60    # 三级对冲


# ─── 数据加载 ────────────────────────────────────────────

def load_csv(filepath):
    df = pd.read_csv(filepath)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
    else:
        df.index = pd.to_datetime(df.index)
    for col in ['Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_stocks(tickers):
    close_dict = {}
    for t in tickers:
        f = STOCK_CACHE / f"{t}.csv"
        if not f.exists() or f.stat().st_size < 500:
            continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 300:
                close_dict[t] = df['Close'].dropna()
        except:
            pass
    return pd.DataFrame(close_dict)


def load_etf(ticker):
    f = CACHE / f"{ticker}.csv"
    if not f.exists():
        return None
    df = load_csv(f)
    return df['Close'].dropna() if 'Close' in df.columns else None


# ─── 选股信号 ────────────────────────────────────────────

def compute_signals(close, i):
    """在第i天计算信号（只用i-1及之前的数据，无前瞻偏差）"""
    hist = close.iloc[:i]  # 截止到i-1
    
    if len(hist) < max(MOM_WINDOW, VOL_WINDOW, SMA_WINDOW) + 5:
        return None, None, None
    
    # 1. 动量信号（6个月，避免最近1个月反转效应）
    # 用 hist.iloc[-MOM_WINDOW-1] 到 hist.iloc[-22]（剔除最近1个月）
    past = hist.iloc[-MOM_WINDOW - 1]
    recent = hist.iloc[-22]  # 剔除最近1个月
    mom = (recent / past - 1).where(past > 0)
    
    # 2. 低波动因子（负值越大越低波动）
    daily_ret = hist.iloc[-VOL_WINDOW:].pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252)
    neg_vol = -vol  # 低波动 = 好
    
    # 3. SMA50趋势过滤（当前价 > SMA50）
    sma50 = hist.iloc[-SMA_WINDOW:].mean()
    curr = hist.iloc[-1]
    above_sma = (curr > sma50)
    
    return mom, neg_vol, above_sma


def select_portfolio(close, i):
    """选择第i天的持仓（用i-1及之前数据）"""
    mom, neg_vol, above_sma = compute_signals(close, i)
    if mom is None:
        return {}
    
    # 标准化
    mom_z = (mom - mom.mean()) / (mom.std() + 1e-9)
    vol_z = (neg_vol - neg_vol.mean()) / (neg_vol.std() + 1e-9)
    
    # 复合因子得分
    score = MOM_WEIGHT * mom_z + VOL_WEIGHT * vol_z
    
    # 只选SMA50以上的股票
    score = score[above_sma]
    
    # 选Top N
    if len(score) == 0:
        return {}
    
    top = score.nlargest(TOP_N)
    
    # 等权（或按得分加权）
    total = top.sum()
    if total <= 0:
        weights = {t: 1.0 / len(top) for t in top.index}
    else:
        weights = {t: max(s, 0) / total for t, s in top.items()}
    
    return weights


# ─── 对冲机制 ────────────────────────────────────────────

def get_hedge_asset(gld_series, tlt_series, i):
    """根据GLD vs TLT的动量选择最优避险资产"""
    if gld_series is None or tlt_series is None:
        return 'GLD'
    
    hist_gld = gld_series.iloc[:i]
    hist_tlt = tlt_series.iloc[:i]
    
    if len(hist_gld) < 63 or len(hist_tlt) < 63:
        return 'GLD'
    
    # 3个月动量对比
    gld_mom = hist_gld.iloc[-1] / hist_gld.iloc[-64] - 1
    tlt_mom = hist_tlt.iloc[-1] / hist_tlt.iloc[-64] - 1
    
    # 选动量更强的避险资产
    return 'GLD' if gld_mom >= tlt_mom else 'TLT'


# ─── 主策略 ──────────────────────────────────────────────

def run_backtest(start='2015-01-01', end='2025-12-31'):
    print(f"📊 Loading stocks from {STOCK_CACHE}...")
    tickers_file = CACHE / "sp500_tickers.txt"
    tickers = tickers_file.read_text().strip().split('\n') if tickers_file.exists() else []
    
    close = load_stocks(tickers + ['SPY'])
    close = close.loc[start:end].ffill()
    
    gld = load_etf('GLD')
    tlt = load_etf('TLT')
    spy = load_etf('SPY')
    
    if gld is not None:
        gld = gld.loc[start:end].ffill()
    if tlt is not None:
        tlt = tlt.loc[start:end].ffill()
    
    print(f"  Loaded {len(close.columns)} stocks, {len(close)} days")
    print(f"  Date range: {close.index[0].date()} to {close.index[-1].date()}")
    
    # 日收益率
    ret = close.pct_change().fillna(0)
    spy_ret = spy.pct_change().fillna(0).loc[start:end] if spy is not None else None
    gld_ret = gld.pct_change().fillna(0) if gld is not None else None
    tlt_ret = tlt.pct_change().fillna(0) if tlt is not None else None
    
    portfolio_ret = pd.Series(0.0, index=close.index)
    equity = pd.Series(1.0, index=close.index)
    
    current_weights = {}
    current_hedge = 0.0
    current_hedge_asset = 'GLD'
    
    rebal_dates = close.index[::REBAL_DAYS]
    
    prev_eq = 1.0
    peak = 1.0
    
    for day_i, date in enumerate(close.index):
        # 每月调仓
        if date in rebal_dates and day_i > max(MOM_WINDOW, VOL_WINDOW, SMA_WINDOW):
            stock_weights = select_portfolio(close, day_i)
            
            # 计算当前回撤
            peak = max(peak, prev_eq)
            dd = prev_eq / peak - 1 if peak > 0 else 0
            
            # 确定对冲比例和资产
            if dd <= DD_THRESHOLD_3:
                hedge_ratio = HEDGE_ALLOC_3
            elif dd <= DD_THRESHOLD_2:
                hedge_ratio = HEDGE_ALLOC_2
            elif dd <= DD_THRESHOLD_1:
                hedge_ratio = HEDGE_ALLOC_1
            else:
                hedge_ratio = 0.0
            
            hedge_asset = get_hedge_asset(gld, tlt, day_i) if hedge_ratio > 0 else 'GLD'
            
            # 最终持仓
            current_weights = {t: w * (1 - hedge_ratio) for t, w in stock_weights.items()}
            current_hedge = hedge_ratio
            current_hedge_asset = hedge_asset
        
        # 计算当日收益
        daily_ret = 0.0
        for ticker, w in current_weights.items():
            if ticker in ret.columns:
                daily_ret += w * ret[ticker].loc[date]
        
        # 对冲资产收益
        if current_hedge > 0:
            hedge_daily = 0.0
            if current_hedge_asset == 'GLD' and gld_ret is not None and date in gld_ret.index:
                hedge_daily = gld_ret.loc[date]
            elif current_hedge_asset == 'TLT' and tlt_ret is not None and date in tlt_ret.index:
                hedge_daily = tlt_ret.loc[date]
            daily_ret += current_hedge * hedge_daily
        
        portfolio_ret[date] = daily_ret
        prev_eq *= (1 + daily_ret)
        equity[date] = prev_eq
    
    return portfolio_ret, equity


# ─── 指标计算 ─────────────────────────────────────────────

def calc_metrics(ret_series, name="v6b"):
    ret = ret_series.dropna()
    cumret = (1 + ret).cumprod()
    n_years = len(ret) / 252
    cagr = cumret.iloc[-1] ** (1 / n_years) - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    rolling_max = cumret.expanding().max()
    drawdown = (cumret / rolling_max) - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {'name': name, 'cagr': round(cagr, 4), 'sharpe': round(sharpe, 4),
            'max_dd': round(max_dd, 4), 'calmar': round(calmar, 4)}


def calc_composite(m):
    return round(0.4 * m['sharpe'] + 0.3 * m['calmar'] + 0.2 * m.get('wf', 0) + 0.1 * (m['cagr'] / 0.3), 4)


def walk_forward(n_splits=4):
    """Walk Forward 验证"""
    import datetime
    start_dt = datetime.date(2015, 1, 1)
    end_dt = datetime.date(2025, 12, 31)
    total_days = (end_dt - start_dt).days
    split_days = total_days // n_splits
    
    is_sharpes, oos_sharpes = [], []
    print(f"\n📐 Walk Forward ({n_splits} splits):")
    
    for i in range(n_splits - 1):
        is_end = start_dt + datetime.timedelta(days=split_days * (i + 1))
        oos_end = start_dt + datetime.timedelta(days=split_days * (i + 2))
        
        is_ret, _ = run_backtest('2015-01-01', is_end.strftime('%Y-%m-%d'))
        oos_ret, _ = run_backtest(is_end.strftime('%Y-%m-%d'), oos_end.strftime('%Y-%m-%d'))
        
        is_m = calc_metrics(is_ret)
        oos_m = calc_metrics(oos_ret)
        is_sharpes.append(is_m['sharpe'])
        oos_sharpes.append(oos_m['sharpe'])
        
        print(f"  Split {i+1}: IS Sharpe={is_m['sharpe']:.3f} | OOS Sharpe={oos_m['sharpe']:.3f}")
    
    avg_is = np.mean(is_sharpes)
    avg_oos = np.mean(oos_sharpes)
    wf = avg_oos / avg_is if avg_is > 0 else 0
    print(f"  Avg IS: {avg_is:.3f}, Avg OOS: {avg_oos:.3f}, WF: {wf:.3f}")
    return wf, avg_is, avg_oos


# ─── 主流程 ──────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("🐻 动量轮动 v6b — 多因子 + TLT/GLD 双对冲")
    print("=" * 60)
    
    print("\n🔁 Full backtest (2015-2025)...")
    ret, equity = run_backtest('2015-01-01', '2025-12-31')
    metrics = calc_metrics(ret, 'v6b')
    
    print(f"\n📊 Full Period Results:")
    print(f"  CAGR:    {metrics['cagr']*100:.1f}%")
    print(f"  Sharpe:  {metrics['sharpe']:.3f}")
    print(f"  MaxDD:   {metrics['max_dd']*100:.1f}%")
    print(f"  Calmar:  {metrics['calmar']:.3f}")
    
    wf, avg_is, avg_oos = walk_forward(4)
    metrics['wf'] = round(wf, 4)
    metrics['composite'] = calc_composite(metrics)
    
    print(f"\n🎯 Composite: {metrics['composite']:.4f}")
    
    v4d = {'cagr': 0.270, 'sharpe': 1.435, 'max_dd': -0.150, 'calmar': 1.805, 'wf': 0.829, 'composite': 1.350}
    print(f"\n📈 vs v4d_base:")
    print(f"  {'Metric':<12} {'v4d':>8} {'v6b':>8} {'Delta':>8}")
    print(f"  {'-'*38}")
    for k in ['cagr', 'sharpe', 'max_dd', 'calmar', 'wf', 'composite']:
        delta = metrics.get(k, 0) - v4d[k]
        print(f"  {k:<12} {v4d[k]:>8.3f} {metrics.get(k,0):>8.3f} {delta:>+8.3f}")
    
    results = {'v4d_base': v4d, 'v6b': metrics}
    out = Path(__file__).parent / 'momentum_v6b_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved to {out}")
    
    c = metrics['composite']
    s = metrics['sharpe']
    if c > 1.8 or s > 2.0:
        print(f"\n🚨 【重大突破】Composite={c:.3f}, Sharpe={s:.3f} 超过阈值！")
    elif c > 1.35:
        print(f"\n✅ 超越v4d！Composite {c:.3f} > 1.350")
    else:
        print(f"\n📝 未超越v4d (Composite {c:.3f} vs 1.350)")
