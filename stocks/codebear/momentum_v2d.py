#!/usr/bin/env python3
"""
动量轮动 v2d — Soft Bear Adaptive (CHAMPION) ⭐
代码熊 🐻

核心创新:
- 在 v2c(v2d_adaptive) 基础上, 将熊市现金比例从 50% 降至 20%
- 保留 80% 仓位在熊市中最强的 Top 8 股票
- 这是 "软着陆" 策略: 不激进减仓, 但通过绝对动量+Vol过滤自然淘汰弱股

策略逻辑:
1. 4因子混合动量: 0.20×1M + 0.40×3M + 0.30×6M + 0.10×12M
2. 绝对动量过滤: 6M return > 0
3. 波动率过滤: 30d annualized vol < 65%
4. 持仓惯性: holdover bonus +3% (减少turnover)
5. 行业分散:
   - Bull (SPY > SMA200): Top 12, max 3/sector, 100% invested
   - Bear (SPY < SMA200): Top 8, max 2/sector, 80% invested + 20% cash
6. Inverse-vol 加权 (风险平价)
7. Monthly rebalance

Results: 
  Full (2015-2025): CAGR 25.8%, Sharpe 1.22, MaxDD -21.9%, Calmar 1.18
  IS (2015-2020):   Sharpe 1.37
  OOS (2021-2025):  Sharpe 1.00
  WF ratio: 0.73 ✅ (target ≥ 0.70)
  Turnover: 48.5%/月 (vs v1 53.0%)
  Composite: 1.013

为什么 "soft bear" 是关键创新:
- v2a/v2b 在熊市全部清仓(100% cash), 错过反弹
- v2c(v2d) 在熊市持有 50% + Top5, 过于保守
- v2d(v2g) 在熊市持有 80% + Top8, 恰好平衡:
  - 2022 年: 绝对动量过滤淘汰科技股, 留下能源/防御股
  - 2022 Q4: 80% 仓位抓住反弹
  - 结果: OOS Sharpe 1.00, 超过 IS 的 0.70 要求
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"


def load_csv(filepath):
    df = pd.read_csv(filepath)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    for col in ['Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_all_data(tickers):
    close_dict, volume_dict = {}, {}
    for t in tickers:
        f = STOCK_CACHE / f"{t}.csv"
        if not f.exists() or f.stat().st_size < 500:
            continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200:
                close_dict[t] = df['Close'].dropna()
                if 'Volume' in df.columns:
                    volume_dict[t] = df['Volume'].dropna()
        except:
            pass
    return pd.DataFrame(close_dict), pd.DataFrame(volume_dict)


def precompute_signals(close_df):
    """Precompute all needed signals vectorized."""
    ret_1m = close_df / close_df.shift(22) - 1
    ret_3m = close_df / close_df.shift(63) - 1
    ret_6m = close_df / close_df.shift(126) - 1
    ret_12m = close_df / close_df.shift(252) - 1
    
    log_ret = np.log(close_df / close_df.shift(1))
    vol_30d = log_ret.rolling(30).std() * np.sqrt(252)
    
    spy_sma200 = close_df['SPY'].rolling(200).mean() if 'SPY' in close_df.columns else None
    
    return {
        'ret_1m': ret_1m, 'ret_3m': ret_3m, 'ret_6m': ret_6m, 'ret_12m': ret_12m,
        'vol_30d': vol_30d, 'spy_sma200': spy_sma200, 'close': close_df,
    }


def get_regime(signals, date):
    spy_sma = signals['spy_sma200']
    if spy_sma is None:
        return 'bull'
    valid = spy_sma.loc[:date].dropna()
    spy_close = signals['close']['SPY'].loc[:date].dropna()
    if len(valid) == 0 or len(spy_close) == 0:
        return 'bull'
    return 'bull' if spy_close.iloc[-1] > valid.iloc[-1] else 'bear'


def select_stocks(signals, sectors, date, prev_holdings):
    """v2d Soft Bear Adaptive selection."""
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}
    idx = idx[-1]
    
    regime = get_regime(signals, date)
    
    # Blended momentum: 0.20×1M + 0.40×3M + 0.30×6M + 0.10×12M
    mom = (signals['ret_1m'].loc[idx] * 0.20 + 
           signals['ret_3m'].loc[idx] * 0.40 +
           signals['ret_6m'].loc[idx] * 0.30 + 
           signals['ret_12m'].loc[idx] * 0.10)
    
    df = pd.DataFrame({
        'momentum': mom,
        'abs_6m': signals['ret_6m'].loc[idx],
        'vol_30d': signals['vol_30d'].loc[idx],
        'price': close.loc[idx],
    })
    
    df = df.dropna(subset=['momentum'])
    df = df[df['price'] >= 5]
    df = df[df.index != 'SPY']
    
    # Absolute momentum + vol filter
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.65)]
    
    # Holdover bonus
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03
    
    df = df.sort_values('momentum', ascending=False)
    
    # Regime-dependent selection
    if regime == 'bull':
        top_n, max_sec, cash = 12, 3, 0.0
    else:
        top_n, max_sec, cash = 8, 2, 0.20  # SOFT BEAR: 80% invested
    
    # Sector-diversified selection
    selected = []
    sector_count = Counter()
    for t in df.index:
        sec = sectors.get(t, 'Unknown')
        if sector_count[sec] < max_sec:
            selected.append(t)
            sector_count[sec] += 1
        if len(selected) >= top_n:
            break
    
    if not selected:
        return {}
    
    # Inverse-vol weighting
    inv_vols = {}
    for t in selected:
        vol = max(df.loc[t, 'vol_30d'], 0.10)
        inv_vols[t] = 1.0 / vol
    total = sum(inv_vols.values())
    invested = 1.0 - cash
    return {t: (v/total) * invested for t, v in inv_vols.items()}


def run_backtest(close_df, signals, sectors, start='2015-01-01', end='2025-12-31',
                 cost_per_trade=0.0015):
    """Monthly backtest."""
    close_range = close_df.loc[start:end].dropna(how='all')
    month_ends = close_range.resample('ME').last().index
    
    portfolio_values, portfolio_dates = [], []
    holdings_history = {}
    turnover_list = []
    prev_weights, prev_holdings = {}, set()
    current_value = 1.0
    
    for i in range(len(month_ends) - 1):
        date, next_date = month_ends[i], month_ends[i + 1]
        
        new_weights = select_stocks(signals, sectors, date, prev_holdings)
        
        # Turnover
        all_t = set(list(new_weights.keys()) + list(prev_weights.keys()))
        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t) / 2
        turnover_list.append(turnover)
        
        prev_weights = new_weights.copy()
        prev_holdings = set(new_weights.keys())
        holdings_history[date.strftime('%Y-%m')] = list(new_weights.keys())
        
        # Period return
        port_ret = sum(
            (close_df[t].loc[date:next_date].dropna().iloc[-1] / 
             close_df[t].loc[date:next_date].dropna().iloc[0] - 1) * w
            for t, w in new_weights.items()
            if len(close_df[t].loc[date:next_date].dropna()) >= 2
        )
        
        port_ret -= turnover * cost_per_trade * 2
        current_value *= (1 + port_ret)
        portfolio_values.append(current_value)
        portfolio_dates.append(next_date)
    
    equity = pd.Series(portfolio_values, index=pd.DatetimeIndex(portfolio_dates))
    return equity, holdings_history, np.mean(turnover_list)


def compute_metrics(equity, name="Strategy"):
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1
    monthly = equity.pct_change().dropna()
    sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0
    dd = (equity - equity.cummax()) / equity.cummax()
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {'name': name, 'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe, 'calmar': calmar}


def main():
    print("=" * 60)
    print("🐻 动量轮动 v2d — Soft Bear Adaptive ⭐")
    print("=" * 60)
    
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df, volume_df = load_all_data(tickers + ['SPY'])
    sectors_file = CACHE / "sp500_sectors.json"
    sectors = json.load(open(sectors_file)) if sectors_file.exists() else {}
    
    signals = precompute_signals(close_df)
    
    # Full
    eq_full, hold_full, to_full = run_backtest(close_df, signals, sectors)
    m_full = compute_metrics(eq_full)
    
    # IS / OOS
    eq_is, _, _ = run_backtest(close_df, signals, sectors, '2015-01-01', '2020-12-31')
    eq_oos, _, _ = run_backtest(close_df, signals, sectors, '2021-01-01', '2025-12-31')
    m_is = compute_metrics(eq_is, "IS")
    m_oos = compute_metrics(eq_oos, "OOS")
    
    wf = m_oos['sharpe'] / m_is['sharpe'] if m_is['sharpe'] != 0 else 0
    
    print(f"\nFull (2015-2025):")
    print(f"  CAGR: {m_full['cagr']:.1%}  Sharpe: {m_full['sharpe']:.2f}  "
          f"MaxDD: {m_full['max_dd']:.1%}  Calmar: {m_full['calmar']:.2f}")
    print(f"\nWalk-Forward:")
    print(f"  IS Sharpe:  {m_is['sharpe']:.2f}")
    print(f"  OOS Sharpe: {m_oos['sharpe']:.2f}")
    print(f"  WF ratio:   {wf:.2f} {'✅' if wf >= 0.70 else '❌'}")
    print(f"\nTurnover: {to_full:.1%}/month")
    
    # Holdings analysis
    hot = {'NVDA','TSLA','META','AVGO','AMD','SMCI','PLTR'}
    print(f"\n2023-2024 Holdings:")
    for ym in sorted(hold_full.keys()):
        if ym.startswith('2023') or ym.startswith('2024'):
            h = [s for s in hold_full[ym] if s in hot]
            print(f"  {ym}: {', '.join(hold_full[ym][:8])} {'🔥'+','.join(h) if h else ''}")
    
    return m_full, m_is, m_oos, hold_full


if __name__ == '__main__':
    main()
