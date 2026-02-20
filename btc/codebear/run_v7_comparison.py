"""
BTC v7 Strategy Comparison — Walk-Forward Validation
代码熊 🐻

Compares v6b baseline, v7a-d variants, and BTC Buy&Hold.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np

BTC_DATA = 'btc/data/btc_daily.csv'
GLD_DATA = 'data_cache/GLD.csv'
START = '2017-01-01'
END = '2026-02-20'
IS_END = '2021-12-31'
OOS_START = '2022-01-01'


def calc_metrics(results, name=''):
    r = results
    if len(r) < 100:
        return None
    years = (r.index[-1] - r.index[0]).days / 365.25
    if years < 0.5:
        return None
    cagr = (r.iloc[-1] / r.iloc[0]) ** (1 / years) - 1
    dd = (r - r.cummax()) / r.cummax()
    max_dd = dd.min()
    dr = r.pct_change().dropna()
    rf = 0.045 / 365
    sharpe = ((dr - rf).mean() / dr.std()) * np.sqrt(365) if dr.std() > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe, 'calmar': calmar}


def composite(sharpe, calmar, cagr, is_sh, oos_sh):
    """Composite = Sharpe*0.4 + Calmar*0.2 + min(CAGR,1.0)*0.2 + WF_bonus(0.2)"""
    wf_pass = oos_sh >= is_sh * 0.7
    return sharpe * 0.4 + calmar * 0.2 + min(cagr, 1.0) * 0.2 + (0.2 if wf_pass else 0)


# ─── Buy & Hold ───
def run_bnh(start, end):
    df = pd.read_csv(BTC_DATA, parse_dates=['Date'], index_col='Date')
    prices = df['Close'].dropna().sort_index().loc[start:end]
    portfolio = (prices / prices.iloc[0]) * 10000
    return portfolio


# ─── V6b Baseline ───
from btc.codebear.beast_v6b import BTCBeastV6b

def run_v6b(start, end):
    bot = BTCBeastV6b()
    bot.load_csv(BTC_DATA, start, end)
    bot.run_backtest()
    return bot.results


# ─── V7a Soft Bear ───
from btc.codebear.beast_v7a import BTCBeastV7a

def run_v7a(start, end):
    bot = BTCBeastV7a()
    bot.load_data(BTC_DATA, start, end)
    bot.run_backtest()
    return bot.results


# ─── V7b GLD Hedge ───
from btc.codebear.beast_v7b import BTCBeastV7b

def run_v7b(start, end):
    bot = BTCBeastV7b()
    bot.load_data(BTC_DATA, GLD_DATA, start, end)
    bot.run_backtest()
    return bot.results


# ─── V7c Soft Bear + GLD ───
from btc.codebear.beast_v7c import BTCBeastV7c

def run_v7c(start, end):
    bot = BTCBeastV7c()
    bot.load_data(BTC_DATA, GLD_DATA, start, end)
    bot.run_backtest()
    return bot.results


# ─── V7d Abs Momentum + GLD ───
from btc.codebear.beast_v7d import BTCBeastV7d

def run_v7d(start, end):
    bot = BTCBeastV7d()
    bot.load_data(BTC_DATA, GLD_DATA, start, end)
    bot.run_backtest()
    return bot.results


# ─── V7e Conservative Momentum + GLD ───
from btc.codebear.beast_v7e import BTCBeastV7e

def run_v7e(start, end):
    bot = BTCBeastV7e()
    bot.load_data(BTC_DATA, GLD_DATA, start, end)
    bot.run_backtest()
    return bot.results


# ─── V7f Dual Momentum ───
from btc.codebear.beast_v7f import BTCBeastV7f

def run_v7f(start, end):
    bot = BTCBeastV7f()
    bot.load_data(BTC_DATA, GLD_DATA, start, end)
    bot.run_backtest()
    return bot.results


# ─── Run All ───
runners = {
    'BTC Buy&Hold': run_bnh,
    'v6b baseline': run_v6b,
    'v7a SoftBear': run_v7a,
    'v7b GLD Hedge': run_v7b,
    'v7c SoftBear+GLD': run_v7c,
    'v7d AbsMom+GLD': run_v7d,
    'v7e ConsvMom+GLD': run_v7e,
    'v7f DualMom': run_v7f,
}

print("=" * 110)
print("BTC Beast v7 Strategy Comparison — 2017-01-01 to 2026-02-20")
print("=" * 110)

rows = []
for name, runner in runners.items():
    try:
        # Full period
        full = runner(START, END)
        m_full = calc_metrics(full)

        # IS
        is_r = runner(START, IS_END)
        m_is = calc_metrics(is_r)

        # OOS
        oos_r = runner(OOS_START, END)
        m_oos = calc_metrics(oos_r)

        if m_full is None or m_is is None or m_oos is None:
            print(f"  ⚠️ {name}: insufficient data, skipping")
            continue

        is_sh = m_is['sharpe']
        oos_sh = m_oos['sharpe']
        wf_pass = oos_sh >= is_sh * 0.7
        comp = composite(m_full['sharpe'], m_full['calmar'], m_full['cagr'], is_sh, oos_sh)

        rows.append({
            'name': name,
            'cagr': m_full['cagr'], 'max_dd': m_full['max_dd'],
            'sharpe': m_full['sharpe'], 'calmar': m_full['calmar'],
            'is_sh': is_sh, 'oos_sh': oos_sh,
            'wf': wf_pass, 'composite': comp,
            'final': full.iloc[-1],
            'oos_maxdd': m_oos['max_dd'],
            'oos_cagr': m_oos['cagr'],
        })

        wf_str = '✅' if wf_pass else '❌'
        print(f"\n{'─'*60}")
        print(f"  {name}")
        print(f"  ${10000:,.0f} → ${full.iloc[-1]:,.0f}")
        print(f"  CAGR: {m_full['cagr']*100:.1f}% | MaxDD: {m_full['max_dd']*100:.1f}% | "
              f"Sharpe: {m_full['sharpe']:.2f} | Calmar: {m_full['calmar']:.2f}")
        print(f"  IS Sharpe: {is_sh:.2f} | OOS Sharpe: {oos_sh:.2f} | "
              f"WF ratio: {oos_sh/is_sh:.2f} | WF: {wf_str}")
        print(f"  OOS MaxDD: {m_oos['max_dd']*100:.1f}% | OOS CAGR: {m_oos['cagr']*100:.1f}%")
        print(f"  Composite: {comp:.3f}")
    except Exception as e:
        print(f"  ❌ {name}: Error — {e}")
        import traceback; traceback.print_exc()

# ─── Summary Table ───
print(f"\n{'='*110}")
header = f"{'版本':<20} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} {'Calmar':>7} {'IS Sh':>7} {'OOS Sh':>7} {'WF ratio':>8} {'WF':>3} {'Composite':>10}"
print(header)
print(f"{'─'*100}")
for r in sorted(rows, key=lambda x: x['composite'], reverse=True):
    wf_str = '✅' if r['wf'] else '❌'
    ratio = r['oos_sh'] / r['is_sh'] if r['is_sh'] != 0 else 0
    print(f"{r['name']:<20} {r['cagr']*100:>6.1f}% {r['max_dd']*100:>6.1f}% "
          f"{r['sharpe']:>7.2f} {r['calmar']:>7.2f} {r['is_sh']:>7.2f} {r['oos_sh']:>7.2f} "
          f"{ratio:>8.2f} {wf_str:>3} {r['composite']:>10.3f}")

# ─── Winner ───
best = max(rows, key=lambda x: x['composite'])
print(f"\n🏆 BEST: {best['name']} (Composite {best['composite']:.3f})")
print(f"   CAGR {best['cagr']*100:.1f}% | MaxDD {best['max_dd']*100:.1f}% | "
      f"Sharpe {best['sharpe']:.2f} | Calmar {best['calmar']:.2f}")
print(f"   WF: {'✅ PASS' if best['wf'] else '❌ FAIL'} "
      f"(IS {best['is_sh']:.2f} → OOS {best['oos_sh']:.2f})")

# ─── 2022 Bear Analysis ───
print(f"\n{'='*110}")
print("2022 Bear Market Analysis (2022-01-01 → 2022-12-31)")
print(f"{'─'*80}")

for name, runner in runners.items():
    try:
        r = runner('2022-01-01', '2022-12-31')
        if r is not None and len(r) > 50:
            ret_2022 = r.iloc[-1] / r.iloc[0] - 1
            dd = (r - r.cummax()) / r.cummax()
            max_dd_2022 = dd.min()
            # Recovery: how long to reach previous high?
            peak_idx = r.cummax().idxmax() if r.idxmax() < r.index[-1] else r.index[0]
            print(f"  {name:<20}: Return {ret_2022*100:>7.1f}% | MaxDD {max_dd_2022*100:>7.1f}%")
    except:
        pass

# ─── v6b comparison ───
v6b_comp = [r for r in rows if r['name'] == 'v6b baseline']
if v6b_comp:
    v6b = v6b_comp[0]
    print(f"\n{'='*110}")
    print(f"vs v6b Baseline (Composite {v6b['composite']:.3f}):")
    for r in rows:
        if r['name'] != 'v6b baseline' and r['name'] != 'BTC Buy&Hold':
            delta = r['composite'] - v6b['composite']
            sign = '+' if delta >= 0 else ''
            print(f"  {r['name']:<20}: {sign}{delta:.3f} "
                  f"(Sharpe {r['sharpe']-v6b['sharpe']:+.2f}, "
                  f"MaxDD {(r['max_dd']-v6b['max_dd'])*100:+.1f}pp, "
                  f"WF {'✅' if r['wf'] else '❌'})")
