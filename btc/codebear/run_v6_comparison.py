"""Run all BTC strategies on 2017+ data with walk-forward validation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np

DATA = 'btc/data/btc_daily.csv'
START = '2017-01-01'
END = '2026-02-20'
IS_END = '2021-12-31'  # 60% IS
OOS_START = '2022-01-01'  # 40% OOS

def calc_metrics(results):
    r = results
    years = (r.index[-1] - r.index[0]).days / 365.25
    cagr = (r.iloc[-1] / r.iloc[0]) ** (1 / years) - 1
    dd = (r - r.cummax()) / r.cummax()
    max_dd = dd.min()
    dr = r.pct_change().dropna()
    rf = 0.045 / 365
    sharpe = ((dr - rf).mean() / dr.std()) * np.sqrt(365) if dr.std() > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return cagr, max_dd, sharpe, calmar

def composite(sharpe, calmar, cagr, is_sh, oos_sh):
    return sharpe * 0.4 + calmar * 0.2 + min(cagr, 1.0) * 0.2 + (0.2 if oos_sh >= is_sh * 0.7 else 0)

# --- V5 ---
from btc.codebear.beast_v5 import BTCBeastV5

def run_v5(start, end):
    bot = BTCBeastV5()
    df = pd.read_csv(DATA, parse_dates=['Date'], index_col='Date')
    df = df[['Close']].dropna().sort_index().loc[start:end]
    bot.data = df
    bot.run_backtest()
    return bot.results

# --- V6a ---
from btc.codebear.beast_v6a import BTCBeastV6a

def run_v6a(start, end):
    bot = BTCBeastV6a()
    bot.load_csv(DATA, start, end)
    bot.run_backtest()
    return bot.results

# --- V6b ---
from btc.codebear.beast_v6b import BTCBeastV6b

def run_v6b(start, end):
    bot = BTCBeastV6b()
    bot.load_csv(DATA, start, end)
    bot.run_backtest()
    return bot.results

# --- V6c ---
from btc.codebear.beast_v6c import BTCBeastV6c

def run_v6c(start, end):
    bot = BTCBeastV6c()
    bot.load_csv(DATA, start, end)
    bot.run_backtest()
    return bot.results

runners = {
    'v5 (2017+)': run_v5,
    'v6a 多周期SMA': run_v6a,
    'v6b 改进减半': run_v6b,
    'v6c vol自适应': run_v6c,
}

print("=" * 100)
print("BTC Beast Strategy Comparison — 2017-01-01 to 2026-02-20")
print("=" * 100)

rows = []
for name, runner in runners.items():
    # Full period
    full = runner(START, END)
    cagr, max_dd, sharpe, calmar = calc_metrics(full)

    # IS
    is_r = runner(START, IS_END)
    _, _, is_sh, _ = calc_metrics(is_r)

    # OOS
    oos_r = runner(OOS_START, END)
    _, _, oos_sh, _ = calc_metrics(oos_r)

    wf_pass = oos_sh >= is_sh * 0.7
    comp = composite(sharpe, calmar, cagr, is_sh, oos_sh)

    rows.append({
        'name': name, 'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe,
        'calmar': calmar, 'is_sh': is_sh, 'oos_sh': oos_sh,
        'wf': '✅' if wf_pass else '❌', 'composite': comp,
        'final': full.iloc[-1]
    })

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"  ${10000:,.0f} → ${full.iloc[-1]:,.0f}")
    print(f"  CAGR: {cagr*100:.1f}% | MaxDD: {max_dd*100:.1f}% | Sharpe: {sharpe:.2f} | Calmar: {calmar:.2f}")
    print(f"  IS Sharpe: {is_sh:.2f} | OOS Sharpe: {oos_sh:.2f} | WF: {'✅' if wf_pass else '❌'}")
    print(f"  Composite: {comp:.3f}")

print(f"\n{'='*100}")
print(f"{'版本':<20} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'IS Sh':>8} {'OOS Sh':>8} {'WF':>4} {'Composite':>10}")
print(f"{'─'*90}")
for r in rows:
    print(f"{r['name']:<20} {r['cagr']*100:>7.1f}% {r['max_dd']*100:>7.1f}% {r['sharpe']:>8.2f} {r['calmar']:>8.2f} {r['is_sh']:>8.2f} {r['oos_sh']:>8.2f} {r['wf']:>4} {r['composite']:>10.3f}")

best = max(rows, key=lambda x: x['composite'])
print(f"\n🏆 Best: {best['name']} (Composite {best['composite']:.3f})")
