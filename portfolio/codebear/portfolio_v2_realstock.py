#!/usr/bin/env python3
"""
Portfolio v2 — Real Stock v3b + BTC v7f Combinations
代码熊 🐻 2026-02-20

Replace QQQ proxy with actual 500-stock momentum strategy.
Test 4 combination methods:
a) Fixed weights (50/50, 60/40, 70/30)
b) Rolling Risk Parity (20-day window)
c) Correlation-adjusted dynamic
d) Target volatility (30% annualized, max 1.5x leverage)

Goal: Composite > 1.50 (vs QQQ proxy 1.488)
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# ═══════════════════════════════════════════════════════════
# PART 1: Generate Stock v3b Daily Returns
# ═══════════════════════════════════════════════════════════

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


def load_all_stock_data(tickers):
    close_dict = {}
    for t in tickers:
        f = STOCK_CACHE / f"{t}.csv"
        if not f.exists() or f.stat().st_size < 500:
            continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200:
                close_dict[t] = df['Close'].dropna()
        except:
            pass
    return pd.DataFrame(close_dict)


def precompute_stock_signals(close_df):
    ret_1m = close_df / close_df.shift(22) - 1
    ret_3m = close_df / close_df.shift(63) - 1
    ret_6m = close_df / close_df.shift(126) - 1
    ret_12m = close_df / close_df.shift(252) - 1
    log_ret = np.log(close_df / close_df.shift(1))
    vol_30d = log_ret.rolling(30).std() * np.sqrt(252)
    spy_close = close_df['SPY'] if 'SPY' in close_df.columns else None
    spy_sma200 = spy_close.rolling(200).mean() if spy_close is not None else None
    sma_50 = close_df.rolling(50).mean()
    return {
        'ret_1m': ret_1m, 'ret_3m': ret_3m, 'ret_6m': ret_6m, 'ret_12m': ret_12m,
        'vol_30d': vol_30d, 'spy_sma200': spy_sma200, 'spy_close': spy_close,
        'sma_50': sma_50, 'close': close_df,
    }


def get_regime(signals, date):
    spy_sma = signals['spy_sma200']
    if spy_sma is None:
        return 'bull'
    valid = spy_sma.loc[:date].dropna()
    spy_close = signals['spy_close'].loc[:date].dropna()
    if len(valid) == 0 or len(spy_close) == 0:
        return 'bull'
    return 'bull' if spy_close.iloc[-1] > valid.iloc[-1] else 'bear'


def select_stocks_v3b(signals, sectors, date, prev_holdings):
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}
    idx = idx[-1]
    regime = get_regime(signals, date)

    mom = (signals['ret_1m'].loc[idx] * 0.20 +
           signals['ret_3m'].loc[idx] * 0.40 +
           signals['ret_6m'].loc[idx] * 0.30 +
           signals['ret_12m'].loc[idx] * 0.10)

    df = pd.DataFrame({
        'momentum': mom, 'abs_6m': signals['ret_6m'].loc[idx],
        'vol_30d': signals['vol_30d'].loc[idx], 'price': close.loc[idx],
        'sma50': signals['sma_50'].loc[idx],
    }).dropna(subset=['momentum', 'sma50'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.65)]
    df = df[df['price'] > df['sma50']]

    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03

    sector_mom = df.groupby('sector')['momentum'].mean().sort_values(ascending=False)

    if regime == 'bull':
        top_sectors = sector_mom.head(4).index.tolist()
        sps = 3
        cash = 0.0
    else:
        top_sectors = sector_mom.head(3).index.tolist()
        sps = 2
        cash = 0.20

    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        selected.extend(sec_df.index[:sps].tolist())

    if not selected:
        return {}

    mom_share = 0.30
    iv = {t: 1.0 / max(df.loc[t, 'vol_30d'], 0.10) for t in selected}
    iv_total = sum(iv.values())
    iv_w = {t: v / iv_total for t, v in iv.items()}

    mom_min = min(df.loc[t, 'momentum'] for t in selected)
    shift = max(-mom_min + 0.01, 0)
    mw = {t: df.loc[t, 'momentum'] + shift for t in selected}
    mw_total = sum(mw.values())
    mw_w = {t: v / mw_total for t, v in mw.items()}

    invested = 1.0 - cash
    return {t: ((1 - mom_share) * iv_w[t] + mom_share * mw_w[t]) * invested for t in selected}


def run_stock_v3b_backtest(close_df, signals, sectors, start='2015-01-01', end='2025-12-31',
                           cost_per_trade=0.0015):
    """Run Stock v3b and return DAILY equity curve (interpolated from monthly rebalance)."""
    close_range = close_df.loc[start:end].dropna(how='all')
    month_ends = close_range.resample('ME').last().index
    
    # We'll track daily returns
    all_daily_returns = {}
    prev_weights, prev_holdings = {}, set()

    for i in range(len(month_ends) - 1):
        date = month_ends[i]
        next_date = month_ends[i + 1]
        new_weights = select_stocks_v3b(signals, sectors, date, prev_holdings)
        
        # Compute turnover cost
        all_t = set(list(new_weights.keys()) + list(prev_weights.keys()))
        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t) / 2
        
        prev_weights = new_weights.copy()
        prev_holdings = set(new_weights.keys())

        # Get daily returns for this period
        period_dates = close_df.loc[date:next_date].index
        if len(period_dates) < 2:
            continue
            
        # Skip the first day (rebalance day)
        for d_idx in range(1, len(period_dates)):
            d = period_dates[d_idx]
            d_prev = period_dates[d_idx - 1]
            
            daily_ret = 0.0
            for t, w in new_weights.items():
                if t in close_df.columns:
                    p_now = close_df[t].get(d, np.nan)
                    p_prev = close_df[t].get(d_prev, np.nan)
                    if not np.isnan(p_now) and not np.isnan(p_prev) and p_prev > 0:
                        daily_ret += w * (p_now / p_prev - 1)
            
            # Apply turnover cost on first day of period
            if d_idx == 1:
                daily_ret -= turnover * cost_per_trade * 2
            
            all_daily_returns[d] = daily_ret

    daily_ret_series = pd.Series(all_daily_returns).sort_index()
    return daily_ret_series


# ═══════════════════════════════════════════════════════════
# PART 2: Generate BTC v7f Daily Returns  
# ═══════════════════════════════════════════════════════════

HALVING_DATES = [
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
]

def halving_info(date, price, prices_series):
    past = [h for h in HALVING_DATES if h <= date]
    if not past:
        return None, None
    last_h = past[-1]
    months = (date - last_h).days / 30.44
    mask = prices_series.index >= last_h
    if mask.any():
        h_price = prices_series.loc[mask].iloc[0]
        gain = (price / h_price) - 1.0
    else:
        gain = 0.0
    return months, gain


def run_btc_v7f_daily(start='2015-01-01', end='2025-12-31'):
    """Run BTC v7f and return daily returns series."""
    btc = pd.read_csv(CACHE / 'BTC_USD.csv', parse_dates=['Date'], index_col='Date')
    btc = btc[['Close']].dropna().sort_index()
    btc.columns = ['BTC']
    
    gld = pd.read_csv(CACHE / 'GLD.csv', parse_dates=['Date'], index_col='Date')
    gld = gld[['Close']].dropna().sort_index()
    gld.columns = ['GLD']
    
    combined = btc.join(gld, how='left')
    combined['GLD'] = combined['GLD'].ffill()
    combined = combined.dropna()
    combined = combined.loc[start:end]
    
    btc_prices = combined['BTC']
    gld_prices = combined['GLD']
    
    btc_mom6 = btc_prices.pct_change(180)
    gld_mom6 = gld_prices.pct_change(180)
    btc_mom3 = btc_prices.pct_change(90)
    gld_mom3 = gld_prices.pct_change(90)
    
    sma200 = btc_prices.rolling(200).mean()
    mayer = btc_prices / sma200
    
    # Track portfolio value for daily returns
    cash = 10000.0
    btc_units = 0.0
    gld_units = 0.0
    portfolio_values = []
    
    for i in range(len(btc_prices)):
        price_btc = btc_prices.iloc[i]
        price_gld = gld_prices.iloc[i]
        mm = mayer.iloc[i]
        b6 = btc_mom6.iloc[i]
        g6 = gld_mom6.iloc[i]
        b3 = btc_mom3.iloc[i]
        g3 = gld_mom3.iloc[i]
        hm, h_gain = halving_info(btc_prices.index[i], price_btc, btc_prices)
        
        # Blended momentum
        if not pd.isna(b6) and not pd.isna(b3):
            btc_mom = 0.5 * b6 + 0.5 * b3
        elif not pd.isna(b6):
            btc_mom = b6
        elif not pd.isna(b3):
            btc_mom = b3
        else:
            btc_mom = 0.05
        
        if not pd.isna(g6) and not pd.isna(g3):
            gld_mom = 0.5 * g6 + 0.5 * g3
        elif not pd.isna(g6):
            gld_mom = g6
        elif not pd.isna(g3):
            gld_mom = g3
        else:
            gld_mom = 0.02
        
        # Dual momentum allocation
        if btc_mom > 0 and gld_mom > 0:
            if btc_mom > gld_mom:
                target_btc = 0.80
                target_gld = 0.15
            else:
                target_btc = 0.50
                target_gld = 0.40
        elif btc_mom > 0 and gld_mom <= 0:
            target_btc = 0.85
            target_gld = 0.05
        elif btc_mom <= 0 and gld_mom > 0:
            target_btc = 0.25
            target_gld = 0.50
        else:
            target_btc = 0.20
            target_gld = 0.30
        
        if hm is not None and hm <= 18:
            target_btc = max(target_btc, 0.50)
            target_gld = min(target_gld, 0.30)
        
        if not pd.isna(mm):
            if mm > 3.5:
                target_btc = min(target_btc, 0.35)
                target_gld = max(target_gld, 0.25)
            elif mm > 3.0:
                target_btc = min(target_btc, 0.50)
                target_gld = max(target_gld, 0.20)
            elif mm > 2.4:
                target_btc = min(target_btc, 0.65)
                target_gld = max(target_gld, 0.15)
        
        if h_gain is not None:
            if h_gain > 5.0:
                target_btc = min(target_btc, 0.35)
                target_gld = max(target_gld, 0.25)
            elif h_gain > 3.0:
                target_btc = min(target_btc, 0.50)
                target_gld = max(target_gld, 0.20)
        
        total = target_btc + target_gld
        if total > 1.0:
            scale = 1.0 / total
            target_btc *= scale
            target_gld *= scale
        
        # Rebalance
        cv = cash + btc_units * price_btc + gld_units * price_gld
        target_btc_val = cv * target_btc
        diff_btc = target_btc_val - btc_units * price_btc
        if diff_btc > 0:
            buy = min(diff_btc, cash)
            btc_units += buy / price_btc
            cash -= buy
        elif diff_btc < 0:
            sell = abs(diff_btc)
            btc_units -= sell / price_btc
            cash += sell
        
        cv = cash + btc_units * price_btc + gld_units * price_gld
        target_gld_val = cv * target_gld
        diff_gld = target_gld_val - gld_units * price_gld
        if diff_gld > 0:
            buy = min(diff_gld, cash)
            gld_units += buy / price_gld
            cash -= buy
        elif diff_gld < 0:
            sell = abs(diff_gld)
            gld_units -= sell / price_gld
            cash += sell
        
        portfolio_values.append(cash + btc_units * price_btc + gld_units * price_gld)
    
    equity = pd.Series(portfolio_values, index=btc_prices.index)
    daily_returns = equity.pct_change().dropna()
    return daily_returns


# ═══════════════════════════════════════════════════════════
# PART 3: Portfolio Combination Methods
# ═══════════════════════════════════════════════════════════

def align_daily_returns(btc_ret, stock_ret):
    """Align two daily return series to common dates."""
    df = pd.DataFrame({'btc': btc_ret, 'stock': stock_ret}).dropna()
    return df['btc'], df['stock']


def fixed_weight_portfolio(btc_ret, stock_ret, w_btc, w_stock):
    """Simple fixed weight combination."""
    btc_a, stock_a = align_daily_returns(btc_ret, stock_ret)
    port_ret = w_btc * btc_a + w_stock * stock_a
    return port_ret


def rolling_risk_parity(btc_ret, stock_ret, lookback=20):
    """Rolling inverse-volatility weighting."""
    btc_a, stock_a = align_daily_returns(btc_ret, stock_ret)
    n = len(btc_a)
    port_ret = pd.Series(0.0, index=btc_a.index)
    
    for i in range(lookback, n):
        btc_vol = btc_a.iloc[i-lookback:i].std() * np.sqrt(252)
        stock_vol = stock_a.iloc[i-lookback:i].std() * np.sqrt(252)
        
        btc_vol = max(btc_vol, 0.01)
        stock_vol = max(stock_vol, 0.01)
        
        inv_btc = 1.0 / btc_vol
        inv_stock = 1.0 / stock_vol
        total_inv = inv_btc + inv_stock
        
        w_btc = inv_btc / total_inv
        w_stock = inv_stock / total_inv
        
        port_ret.iloc[i] = w_btc * btc_a.iloc[i] + w_stock * stock_a.iloc[i]
    
    return port_ret.iloc[lookback:]


def correlation_adjusted_portfolio(btc_ret, stock_ret, lookback=60, base_btc=0.50, base_stock=0.50):
    """Dynamic correlation-adjusted allocation."""
    btc_a, stock_a = align_daily_returns(btc_ret, stock_ret)
    n = len(btc_a)
    port_ret = pd.Series(0.0, index=btc_a.index)
    
    for i in range(lookback, n):
        window_btc = btc_a.iloc[i-lookback:i]
        window_stock = stock_a.iloc[i-lookback:i]
        
        corr = window_btc.corr(window_stock)
        if np.isnan(corr):
            corr = 0.0
        
        # Inverse vol base weights
        btc_vol = window_btc.std() * np.sqrt(252)
        stock_vol = window_stock.std() * np.sqrt(252)
        btc_vol = max(btc_vol, 0.01)
        stock_vol = max(stock_vol, 0.01)
        
        inv_btc = 1.0 / btc_vol
        inv_stock = 1.0 / stock_vol
        total_inv = inv_btc + inv_stock
        w_btc = inv_btc / total_inv
        w_stock = inv_stock / total_inv
        
        # Correlation adjustment
        if corr > 0.5:
            # High correlation → reduce total exposure (more cash-like)
            scale = 1.0 - 0.3 * (corr - 0.5)  # reduce up to 15% when corr=1.0
            scale = max(scale, 0.70)
            w_btc *= scale
            w_stock *= scale
        elif corr < 0:
            # Negative correlation → increase exposure (diversification benefit)
            boost = 1.0 + 0.2 * abs(corr)  # boost up to 20% when corr=-1.0
            boost = min(boost, 1.20)
            w_btc *= boost
            w_stock *= boost
        
        # Normalize so weights sum to ≤ 1.0
        total = w_btc + w_stock
        if total > 1.0:
            w_btc /= total
            w_stock /= total
        
        port_ret.iloc[i] = w_btc * btc_a.iloc[i] + w_stock * stock_a.iloc[i]
    
    return port_ret.iloc[lookback:]


def target_volatility_portfolio(btc_ret, stock_ret, target_vol=0.30, lookback=20, max_leverage=1.5):
    """Target volatility portfolio with dynamic leverage."""
    btc_a, stock_a = align_daily_returns(btc_ret, stock_ret)
    n = len(btc_a)
    port_ret = pd.Series(0.0, index=btc_a.index)
    
    for i in range(lookback, n):
        # First compute risk-parity base weights
        btc_vol = btc_a.iloc[i-lookback:i].std() * np.sqrt(252)
        stock_vol = stock_a.iloc[i-lookback:i].std() * np.sqrt(252)
        
        btc_vol = max(btc_vol, 0.01)
        stock_vol = max(stock_vol, 0.01)
        
        inv_btc = 1.0 / btc_vol
        inv_stock = 1.0 / stock_vol
        total_inv = inv_btc + inv_stock
        w_btc = inv_btc / total_inv
        w_stock = inv_stock / total_inv
        
        # Compute portfolio vol with these weights and correlation
        window_btc = btc_a.iloc[i-lookback:i]
        window_stock = stock_a.iloc[i-lookback:i]
        corr = window_btc.corr(window_stock)
        if np.isnan(corr):
            corr = 0.0
        
        port_vol = np.sqrt(
            (w_btc * btc_vol)**2 + 
            (w_stock * stock_vol)**2 + 
            2 * w_btc * w_stock * btc_vol * stock_vol * corr
        )
        
        # Leverage to hit target vol
        if port_vol > 0:
            leverage = target_vol / port_vol
        else:
            leverage = 1.0
        
        leverage = min(leverage, max_leverage)
        leverage = max(leverage, 0.3)  # minimum 30% invested
        
        w_btc_final = w_btc * leverage
        w_stock_final = w_stock * leverage
        
        port_ret.iloc[i] = w_btc_final * btc_a.iloc[i] + w_stock_final * stock_a.iloc[i]
    
    return port_ret.iloc[lookback:]


# ═══════════════════════════════════════════════════════════
# PART 4: Metrics & Walk-Forward
# ═══════════════════════════════════════════════════════════

def calc_metrics(returns, rf_annual=0.04):
    """Calculate comprehensive strategy metrics from daily returns."""
    returns = returns.dropna()
    if len(returns) < 126:
        return None
    
    cum = (1 + returns).cumprod()
    years = len(returns) / 252
    final_val = cum.iloc[-1]
    cagr = final_val ** (1 / years) - 1
    
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    
    daily_rf = (1 + rf_annual) ** (1/252) - 1
    excess = returns - daily_rf
    sharpe = np.sqrt(252) * excess.mean() / (excess.std() + 1e-10)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    ann_vol = returns.std() * np.sqrt(252)
    
    # Monthly returns for positive month count
    monthly = (1 + returns).resample('ME').prod() - 1
    pos_months = (monthly > 0).sum()
    total_months = len(monthly)
    
    return {
        'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe, 'calmar': calmar,
        'ann_vol': ann_vol, 'final_val': final_val, 'years': years,
        'pos_months': pos_months, 'total_months': total_months,
    }


def composite_score(m):
    """Composite = Sharpe×0.4 + Calmar×0.4 + CAGR×0.2"""
    return m['sharpe'] * 0.4 + m['calmar'] * 0.4 + m['cagr'] * 0.2


def walk_forward(returns, split_date='2020-12-31'):
    """Walk-forward: IS before split, OOS after split."""
    is_ret = returns.loc[:split_date]
    oos_ret = returns.loc[split_date:]
    # Remove first day of OOS to avoid overlap
    if len(oos_ret) > 0:
        oos_ret = oos_ret.iloc[1:]
    
    is_m = calc_metrics(is_ret)
    oos_m = calc_metrics(oos_ret)
    
    if is_m is None or oos_m is None:
        return None, None, 0.0
    
    wf = oos_m['sharpe'] / is_m['sharpe'] if is_m['sharpe'] > 0 else 0
    return is_m, oos_m, wf


def yearly_returns(returns):
    """Compute annual returns."""
    yearly = {}
    for year in range(2015, 2026):
        year_ret = returns.loc[f'{year}-01-01':f'{year}-12-31']
        if len(year_ret) > 20:
            cum = (1 + year_ret).prod() - 1
            yearly[year] = cum
        else:
            yearly[year] = np.nan
    return yearly


def drawdown_analysis(returns, periods):
    """Analyze drawdowns in specific periods."""
    results = {}
    cum = (1 + returns).cumprod()
    
    for name, (start, end) in periods.items():
        period_cum = cum.loc[start:end]
        if len(period_cum) < 5:
            results[name] = {'max_dd': np.nan, 'recovery_days': np.nan}
            continue
        
        peak = period_cum.cummax()
        dd = (period_cum - peak) / peak
        max_dd = dd.min()
        
        # Recovery: days from trough to new high
        trough_idx = dd.idxmin()
        post_trough = cum.loc[trough_idx:]
        pre_trough_peak = peak.loc[trough_idx]
        recovered = post_trough[post_trough >= pre_trough_peak]
        if len(recovered) > 0:
            recovery_days = (recovered.index[0] - trough_idx).days
        else:
            recovery_days = -1  # not recovered
        
        results[name] = {'max_dd': max_dd, 'recovery_days': recovery_days}
    
    return results


# ═══════════════════════════════════════════════════════════
# PART 5: Run Traditional 60/40 Benchmark
# ═══════════════════════════════════════════════════════════

def run_traditional_6040(start='2015-01-01', end='2025-12-31'):
    """SPY 60% + GLD 40% (using GLD as bond proxy since AGG not available)."""
    spy = pd.read_csv(CACHE / 'SPY.csv', parse_dates=['Date'], index_col='Date')['Close'].dropna().sort_index()
    
    # Try TLT as bond proxy
    tlt_path = CACHE / 'TLT.csv'
    if tlt_path.exists():
        bond = pd.read_csv(tlt_path, parse_dates=['Date'], index_col='Date')['Close'].dropna().sort_index()
    else:
        bond = pd.read_csv(CACHE / 'GLD.csv', parse_dates=['Date'], index_col='Date')['Close'].dropna().sort_index()
    
    df = pd.DataFrame({'SPY': spy, 'BOND': bond}).dropna().loc[start:end]
    spy_ret = df['SPY'].pct_change().dropna()
    bond_ret = df['BOND'].pct_change().dropna()
    
    port_ret = 0.60 * spy_ret + 0.40 * bond_ret
    return port_ret


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("🐻 Portfolio v2 — Real Stock v3b + BTC v7f Combinations")
    print("  Goal: Composite > 1.50 (vs QQQ proxy 1.488)")
    print("=" * 80)

    # ── Step 1: Generate Stock v3b daily returns ──
    print("\n[1] Running Stock v3b (500-stock momentum rotation)...")
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_all_stock_data(tickers + ['SPY'])
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    signals = precompute_stock_signals(close_df)
    
    stock_daily = run_stock_v3b_backtest(close_df, signals, sectors)
    print(f"  Stock v3b: {len(stock_daily)} daily returns, {stock_daily.index[0].date()} → {stock_daily.index[-1].date()}")
    
    # Save
    stock_csv = BASE / "stocks" / "codebear" / "v3b_daily_returns.csv"
    pd.DataFrame({'Date': stock_daily.index, 'Return': stock_daily.values}).to_csv(stock_csv, index=False)
    print(f"  Saved to {stock_csv}")

    # ── Step 2: Generate BTC v7f daily returns ──
    print("\n[2] Running BTC v7f (dual momentum rotation, 2015-2025)...")
    btc_daily = run_btc_v7f_daily(start='2015-01-01', end='2025-12-31')
    print(f"  BTC v7f: {len(btc_daily)} daily returns, {btc_daily.index[0].date()} → {btc_daily.index[-1].date()}")
    
    # Save
    btc_csv = BASE / "btc" / "codebear" / "v7f_daily_returns_2015_2025.csv"
    pd.DataFrame({'Date': btc_daily.index, 'Return': btc_daily.values}).to_csv(btc_csv, index=False)
    print(f"  Saved to {btc_csv}")

    # ── Step 3: Individual strategy metrics ──
    print("\n[3] Individual Strategy Metrics:")
    stock_m = calc_metrics(stock_daily)
    btc_m = calc_metrics(btc_daily)
    
    print(f"  Stock v3b: CAGR={stock_m['cagr']:.1%} MaxDD={stock_m['max_dd']:.1%} Sharpe={stock_m['sharpe']:.2f} Calmar={stock_m['calmar']:.2f} Comp={composite_score(stock_m):.3f}")
    print(f"  BTC v7f:   CAGR={btc_m['cagr']:.1%} MaxDD={btc_m['max_dd']:.1%} Sharpe={btc_m['sharpe']:.2f} Calmar={btc_m['calmar']:.2f} Comp={composite_score(btc_m):.3f}")
    
    # Correlation
    btc_a, stock_a = align_daily_returns(btc_daily, stock_daily)
    overall_corr = btc_a.corr(stock_a)
    print(f"\n  BTC/Stock correlation: {overall_corr:.3f}")

    # ── Step 4: Build Portfolio Combinations ──
    print("\n[4] Portfolio Combinations:")
    print(f"{'Strategy':<25} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} {'Calmar':>7} {'Vol':>6} {'IS Sh':>6} {'OOS Sh':>7} {'WF':>5} {'Comp':>7}")
    print("-" * 100)

    portfolios = {}
    
    # a) Fixed weights
    for name, wb, ws in [('50/50 Fixed', 0.50, 0.50), ('60/40 BTC/Stock', 0.60, 0.40), ('70/30 BTC/Stock', 0.70, 0.30)]:
        ret = fixed_weight_portfolio(btc_daily, stock_daily, wb, ws)
        portfolios[name] = ret
    
    # b) Rolling Risk Parity (20-day)
    ret = rolling_risk_parity(btc_daily, stock_daily, lookback=20)
    portfolios['Rolling RP 20d'] = ret
    
    # Also test 60-day and 126-day lookbacks
    ret60 = rolling_risk_parity(btc_daily, stock_daily, lookback=60)
    portfolios['Rolling RP 60d'] = ret60
    
    # c) Correlation-adjusted
    ret_corr = correlation_adjusted_portfolio(btc_daily, stock_daily, lookback=60)
    portfolios['Corr-Adjusted'] = ret_corr
    
    # d) Target volatility
    ret_tvol = target_volatility_portfolio(btc_daily, stock_daily, target_vol=0.30, lookback=20)
    portfolios['Target Vol 30%'] = ret_tvol
    
    # Individual strategies for comparison
    portfolios['Stock v3b Solo'] = stock_daily
    portfolios['BTC v7f Solo'] = btc_daily
    
    # Traditional 60/40
    trad = run_traditional_6040()
    portfolios['Trad SPY60/TLT40'] = trad

    # Print results
    results = {}
    for name, ret in portfolios.items():
        m = calc_metrics(ret)
        if m is None:
            continue
        is_m, oos_m, wf = walk_forward(ret)
        comp = composite_score(m)
        
        is_sh = is_m['sharpe'] if is_m else 0
        oos_sh = oos_m['sharpe'] if oos_m else 0
        wf_mark = '✅' if wf >= 0.70 else '❌'
        
        results[name] = {
            'metrics': m, 'is_m': is_m, 'oos_m': oos_m, 'wf': wf, 'composite': comp,
        }
        
        print(f"{name:<25} {m['cagr']:>6.1%} {m['max_dd']:>6.1%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {m['ann_vol']:>5.1%} {is_sh:>6.2f} {oos_sh:>7.2f} {wf:>4.2f}{wf_mark} {comp:>7.3f}")

    # ── Step 5: Best Portfolio Selection ──
    print("\n" + "=" * 80)
    print("[5] 🏆 BEST PORTFOLIO SELECTION")
    print("=" * 80)
    
    # Filter WF-passing
    wf_passing = {k: v for k, v in results.items() 
                  if v['wf'] >= 0.70 and 'Solo' not in k and 'Trad' not in k}
    
    if wf_passing:
        best_name = max(wf_passing, key=lambda k: wf_passing[k]['composite'])
        best = wf_passing[best_name]
        bm = best['metrics']
        
        print(f"\n  🏆 CHAMPION: {best_name}")
        print(f"     CAGR:      {bm['cagr']:.1%}")
        print(f"     MaxDD:     {bm['max_dd']:.1%}")
        print(f"     Sharpe:    {bm['sharpe']:.2f}")
        print(f"     Calmar:    {bm['calmar']:.2f}")
        print(f"     Ann Vol:   {bm['ann_vol']:.1%}")
        print(f"     IS Sharpe: {best['is_m']['sharpe']:.2f}")
        print(f"     OOS Sharpe:{best['oos_m']['sharpe']:.2f}")
        print(f"     WF Ratio:  {best['wf']:.2f} ✅")
        print(f"     Composite: {best['composite']:.3f}")
        print(f"     vs QQQ proxy (1.488): {'BETTER ✅' if best['composite'] > 1.488 else 'WORSE ❌'}")
        print(f"     vs 1.50 target: {'ACHIEVED ✅' if best['composite'] > 1.50 else 'NOT YET ❌'}")
    else:
        print("\n  ❌ No portfolio passed Walk-Forward validation")
        best_name = max(results, key=lambda k: results[k]['composite'])
        best = results[best_name]
        bm = best['metrics']
        print(f"  Best overall: {best_name} Composite={best['composite']:.3f} WF={best['wf']:.2f}")

    # ── Step 6: Yearly Performance ──
    print("\n" + "=" * 80)
    print("[6] Yearly Performance")
    print("=" * 80)
    
    key_strats = ['BTC v7f Solo', 'Stock v3b Solo', 'Rolling RP 20d', '50/50 Fixed', 'Corr-Adjusted', 'Target Vol 30%']
    key_strats = [s for s in key_strats if s in portfolios]
    
    print(f"\n{'Year':<6}", end='')
    for s in key_strats:
        print(f" {s[:15]:>15}", end='')
    print(f" {'Best':>15}")
    print("-" * (6 + 16 * (len(key_strats) + 1)))
    
    for year in range(2015, 2026):
        print(f"{year:<6}", end='')
        yr_rets = {}
        for s in key_strats:
            yr = yearly_returns(portfolios[s])
            val = yr.get(year, np.nan)
            yr_rets[s] = val
            if np.isnan(val):
                print(f" {'N/A':>15}", end='')
            else:
                print(f" {val:>14.1%}", end='')
        
        # Best
        valid = {k: v for k, v in yr_rets.items() if not np.isnan(v)}
        if valid:
            best_yr = max(valid, key=valid.get)
            print(f" {best_yr[:15]:>15}", end='')
        print()

    # ── Step 7: Drawdown Analysis ──
    print("\n" + "=" * 80)
    print("[7] Drawdown Analysis (Key Crisis Periods)")
    print("=" * 80)
    
    crisis_periods = {
        '2018 Crypto Bear': ('2018-01-01', '2018-12-31'),
        '2020 COVID Crash': ('2020-02-01', '2020-04-30'),
        '2022 Bear Market': ('2022-01-01', '2022-12-31'),
    }
    
    print(f"\n{'Period':<22}", end='')
    for s in key_strats:
        print(f" {s[:12]:>14}", end='')
    print()
    print("-" * (22 + 15 * len(key_strats)))
    
    for period_name, (start, end) in crisis_periods.items():
        print(f"{period_name:<22}", end='')
        for s in key_strats:
            da = drawdown_analysis(portfolios[s], {period_name: (start, end)})
            dd = da[period_name]['max_dd']
            if np.isnan(dd):
                print(f" {'N/A':>14}", end='')
            else:
                print(f" {dd:>13.1%}", end='')
        print()

    # ── Step 8: $10k Investment Growth ──
    print("\n" + "=" * 80)
    print("[8] $10,000 Initial Investment → Final Value")
    print("=" * 80)
    
    for s in key_strats:
        m = results.get(s, {}).get('metrics')
        if m:
            final = 10000 * m['final_val']
            print(f"  {s:<25}: ${final:>12,.0f}  ({m['cagr']:.1%} CAGR over {m['years']:.1f}yr)")

    # ── Step 9: Diversification Benefit ──
    print("\n" + "=" * 80)
    print("[9] Diversification Benefit Analysis")
    print("=" * 80)
    
    btc_a2, stock_a2 = align_daily_returns(btc_daily, stock_daily)
    btc_cagr = btc_m['cagr'] if btc_m else 0
    stock_cagr = stock_m['cagr'] if stock_m else 0
    
    for name in ['50/50 Fixed', '60/40 BTC/Stock', 'Rolling RP 20d', 'Corr-Adjusted', 'Target Vol 30%']:
        if name in results:
            port_cagr = results[name]['metrics']['cagr']
            if '50/50' in name:
                weighted_avg = 0.50 * btc_cagr + 0.50 * stock_cagr
            elif '60/40' in name:
                weighted_avg = 0.60 * btc_cagr + 0.40 * stock_cagr
            elif '70/30' in name:
                weighted_avg = 0.70 * btc_cagr + 0.30 * stock_cagr
            else:
                weighted_avg = 0.50 * btc_cagr + 0.50 * stock_cagr  # approximate
            
            div_benefit = port_cagr - weighted_avg
            print(f"  {name:<25}: Portfolio CAGR={port_cagr:.1%}, Weighted Avg={weighted_avg:.1%}, Diversification={div_benefit:+.1%}")

    # ── Step 10: Rolling Correlation ──
    print("\n" + "=" * 80)
    print("[10] Rolling Correlation (BTC vs Stock)")
    print("=" * 80)
    
    rolling_corr = btc_a2.rolling(60).corr(stock_a2)
    for year in range(2015, 2026):
        yr_corr = rolling_corr.loc[f'{year}-01-01':f'{year}-12-31']
        if len(yr_corr) > 0:
            print(f"  {year}: avg={yr_corr.mean():.3f}, min={yr_corr.min():.3f}, max={yr_corr.max():.3f}")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    if wf_passing:
        best_comp = best['composite']
        print(f"\n  🏆 Best Portfolio: {best_name}")
        print(f"     Composite Score: {best_comp:.3f}")
        print(f"     vs QQQ proxy (1.488): {best_comp - 1.488:+.3f}")
        print(f"     vs 1.50 target: {best_comp - 1.50:+.3f}")
        print(f"     Recommendation: {'UPDATE BEST_STRATEGY.md ✅' if best_comp > 1.488 else 'Keep current ❌'}")
    
    # Show all WF-passing sorted
    print(f"\n  All WF-passing portfolios (sorted by Composite):")
    for name, data in sorted(wf_passing.items(), key=lambda x: x[1]['composite'], reverse=True):
        m = data['metrics']
        print(f"    {name:<25} Comp={data['composite']:.3f} Sharpe={m['sharpe']:.2f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} WF={data['wf']:.2f}")

    print("\n🐻 Portfolio v2 Analysis Complete!")
    return results, portfolios


if __name__ == '__main__':
    results, portfolios = main()
