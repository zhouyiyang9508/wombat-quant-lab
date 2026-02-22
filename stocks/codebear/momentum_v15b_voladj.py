"""
Momentum v15b — Vol-Adjusted 动量实验
=======================================
v11b 的动量信号 = r1*w1 + r3*w3 + r6*w6 + r12*w12

Vol-Adjusted 版本：
  将各期收益除以对应窗口的年化波动率
  等价于每只股票在该时间窗口的 Sharpe ratio

  VA = (r_N / vol_N) 代替 r_N
  
其中 vol_N = 过去 N 月日收益率的年化标准差

代码熊 🐻 | 2026-02-22
"""
import pandas as pd, numpy as np, json, copy, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / 'data_cache'
sys.path.insert(0, str(BASE))
import stocks.codebear.momentum_v11b_final as v11b

def lc(n):
    fp = CACHE / f'{n}.csv'; df = pd.read_csv(fp)
    c = 'Date' if 'Date' in df.columns else df.columns[0]
    df[c] = pd.to_datetime(df[c])
    return df.set_index(c).sort_index()['Close'].dropna()

def wf_score(eq, is_end='2021-12-31', oos_start='2022-01-01', rf=0.04):
    ie = eq[eq.index <= pd.Timestamp(is_end)]
    oe = eq[eq.index >= pd.Timestamp(oos_start)]
    if len(ie) < 200 or len(oe) < 100: return 0, 0, 0
    def sh(e):
        r = e.pct_change().dropna(); n = len(r)
        return float(((e.iloc[-1]/e.iloc[0])**(252/n)-1-rf) / (r.std()*np.sqrt(252))) if r.std()>0 else 0
    i, o = sh(ie), sh(oe)
    return round(o/i,3) if i>0 else 0, round(i,3), round(o,3)

def make_voladj_sig(sig, close_df, blend=0.0):
    """
    复制 sig 并用 vol-adjusted 版本替换 r1/r3/r6/r12
    blend: 0=纯 vol-adj, 1=纯原始, 0.5=混合
    """
    new_sig = dict(sig)  # shallow copy，r1/r3/r6/r12 会被替换

    for key, days in [('r1', 21), ('r3', 63), ('r6', 126), ('r12', 252)]:
        orig_df = sig[key]  # index=dates, columns=tickers, val=return
        
        # 计算每个日期对应的 N 天年化波动率
        # vol(date, ticker) = std(daily_ret over past days) * sqrt(252)
        vol_df = close_df.pct_change().rolling(days, min_periods=max(10, days//3)).std() * np.sqrt(252)
        vol_df = vol_df.reindex(orig_df.index).ffill()
        
        # Vol-adjusted return
        with np.errstate(divide='ignore', invalid='ignore'):
            va_df = orig_df / vol_df.clip(lower=0.01)  # 防止除以接近 0 的波动率
        va_df = va_df.replace([np.inf, -np.inf], np.nan)
        
        if blend > 0:
            # 线性混合：用 zscore 统一量纲后加权
            def zscore(df):
                m = df.mean(axis=1); s = df.std(axis=1)
                return df.sub(m, axis=0).div(s.clip(lower=0.01), axis=0)
            new_sig[key] = blend * zscore(orig_df) + (1-blend) * zscore(va_df)
        else:
            new_sig[key] = va_df
    
    return new_sig

if __name__ == '__main__':
    print("=" * 65)
    print("🐻 v15b — Vol-Adjusted 动量实验")
    print("=" * 65)

    print("\n[1/2] 加载数据...")
    sp500_tickers = (CACHE/"sp500_tickers.txt").read_text().strip().split('\n')
    sectors = json.load(open(CACHE/"sp500_sectors.json"))
    gld_p  = lc('GLD'); gdx_p = lc('GDX'); gdxj_p = lc('GDXJ')
    shy_p  = lc('SHY'); tlt_p = lc('TLT'); ief_p  = lc('IEF')
    def_prices = {'XLV':lc('XLV'), 'XLP':lc('XLP'), 'XLU':lc('XLU')}

    close_df = v11b.load_stocks(sp500_tickers + ['SPY'])
    close_df = close_df.loc['2014-01-01':'2025-12-31']
    print(f"  {close_df.shape[1]} tickers, {len(close_df)} trading days")

    print("[2/2] 预计算信号...", flush=True)
    sig_orig = v11b.precompute(close_df)

    close_bt = close_df.loc['2015-01-01':'2025-12-31']

    configs = [
        (None, "📊 Baseline — 纯原始动量"),
        (0.0,  "📊 VA-Pure — 纯 vol-adjusted (mom/vol)"),
        (0.5,  "📊 VA-Blend50 — 50% 原始 + 50% vol-adj"),
        (0.3,  "📊 VA-Blend30 — 30% 原始 + 70% vol-adj"),
        (0.7,  "📊 VA-Blend70 — 70% 原始 + 30% vol-adj"),
    ]

    results = []
    for blend, label in configs:
        try:
            sig_use = sig_orig if blend is None else make_voladj_sig(sig_orig, close_df, blend=blend)
            eq, avg_to, _, _ = v11b.run_backtest(
                close_bt, sig_use, sectors,
                gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices)
            m = v11b.compute_metrics(eq)
            wfr, is_sh, oos_sh = wf_score(eq)
            comp = m['sharpe']*0.4 + m['calmar']*0.4 + min(m['cagr'],1.0)*0.2
            flag = '✅' if wfr>=0.60 else '⚠️' if wfr>=0.55 else '❌'
            print(f"\n{label}")
            print(f"  CAGR={m['cagr']*100:.1f}%  MaxDD={m['max_dd']*100:.1f}%  "
                  f"Sh={m['sharpe']:.2f}  Cal={m['calmar']:.2f}  Comp={comp:.3f}  "
                  f"WF={wfr}{flag}  IS={is_sh:.2f}→OOS={oos_sh:.2f}  TO={avg_to*100:.1f}%")
            results.append((label, m, wfr, is_sh, oos_sh, comp))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"\n{label}: ERROR — {e}")

    print("\n" + "=" * 65)
    print("📋 汇总对比")
    print("=" * 65)
    base = next((r for r in results if 'Baseline' in r[0]), None)
    for label, m, wfr, is_sh, oos_sh, comp in results:
        flag = '✅' if wfr>=0.60 else '⚠️'
        dc = f"({comp-(base[-1] if base else 0):+.3f})" if base and 'Baseline' not in label else ""
        print(f"  {label[2:]:<35} CAGR={m['cagr']*100:.1f}%  Comp={comp:.3f}{dc}  WF={wfr}{flag}")

    if results:
        best = max(results, key=lambda x: x[-1])
        print(f"\n🏆 最优: {best[0]}  Comp={best[-1]:.3f}")
