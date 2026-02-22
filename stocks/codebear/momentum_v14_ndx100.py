"""
Momentum v14 — Nasdaq-100 宇宙实验
=====================================
研究问题：把 S&P 500 换成 Nasdaq-100（NDX），CAGR 能提升多少？

NDX 特点：
  - 100 只股票（高度集中）
  - 科技占约 50%，无金融板块
  - 历史上 CAGR > S&P500
  - 动量因子在成长股上更强

实验矩阵：
  A. NDX 全量（100只，但行业极度不均衡）
  B. NDX + GLD/GDXJ 对冲保持不变
  C. NDX + SPY 作为宽度指标（用 SPY 动量判断牛熊，但选股从 NDX 里选）

代码熊 🐻 | 2026-02-22
"""
import pandas as pd, numpy as np, json, sys, time, urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / 'data_cache'
STOCKS = CACHE / 'stocks'
sys.path.insert(0, str(BASE))
import stocks.codebear.momentum_v11b_final as v11b

# ─── Nasdaq-100 成分股（2024年版，含行业分类）────────────────────────────────
NDX100 = {
    # Technology (~50%)
    'Technology': [
        'AAPL', 'MSFT', 'NVDA', 'AVGO', 'AMD', 'ADBE', 'ASML', 'CSCO',
        'QCOM', 'INTC', 'INTU', 'AMAT', 'TXN', 'MU', 'KLAC', 'SNPS',
        'LRCX', 'CDNS', 'MCHP', 'ADI', 'NXPI', 'ON', 'MRVL', 'ANSS',
        'FTNT', 'PANW', 'CRWD', 'ZS', 'DDOG', 'CTSH', 'CDW', 'SMCI',
        'ARM', 'GDDY', 'FFIV',
    ],
    # ConsumerDisc (~15%)
    'ConsumerDisc': [
        'AMZN', 'TSLA', 'COST', 'SBUX', 'ORLY', 'ROST', 'ODFL',
        'CPRT', 'DLTR', 'TTWO', 'EA', 'ABNB', 'DASH', 'EBAY', 'LULU',
        'PCAR', 'FAST',
    ],
    # CommServices (~10%)
    'CommServices': [
        'META', 'GOOGL', 'GOOG', 'NFLX', 'CMCSA', 'CHTR',
        'WBD', 'MTCH', 'NTES', 'BIDU',
    ],
    # Healthcare (~8%)
    'Healthcare': [
        'ISRG', 'REGN', 'GILD', 'VRTX', 'BIIB', 'DXCM',
        'IDXX', 'ILMN', 'GEHC', 'MRNA',
    ],
    # Industrials (~6%)
    'Industrials': [
        'ADP', 'PAYX', 'CTAS', 'CSX', 'BKR', 'VRSK', 'ROP',
    ],
    # ConsumerStap (~6%)
    'ConsumerStap': [
        'PEP', 'MNST', 'KDP', 'KHC', 'WBA',
    ],
    # Utilities (~2%)
    'Utilities': [
        'CEG', 'XEL', 'EXC',
    ],
    # Other/Mixed（放 Technology 不合适的）
    'Materials': [
        'FANG',  # 能源，但 NDX 无能源板块，放 Materials
    ],
    'RealEstate': [
        'CSGP',  # CoStar
    ],
}

def stooq_download(ticker, out_path):
    sym = ticker.lower().replace('-', '.') + '.us'
    url = f'https://stooq.com/q/d/l/?s={sym}&d1=20140601&d2=20251231&i=d'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0)'})
        data = urllib.request.urlopen(req, timeout=15).read().decode()
        lines = data.strip().split('\n')
        if len(lines) > 200 and 'Date' in lines[0]:
            out_path.write_text(data); return True
        return False
    except Exception:
        return False

def load_close(tickers):
    frames = {}
    for t in tickers:
        f = STOCKS / f'{t}.csv'
        if not f.exists(): continue
        df = pd.read_csv(f)
        col = 'Date' if 'Date' in df.columns else df.columns[0]
        df[col] = pd.to_datetime(df[col])
        df = df.set_index(col).sort_index()
        if 'Close' not in df.columns: continue
        s = df['Close'].dropna()
        if len(s) < 500: continue
        frames[t] = s
    return pd.DataFrame(frames) if frames else pd.DataFrame()

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
        return float((r.mean()-rf/12)/r.std()*np.sqrt(12)) if r.std()>0 else 0
    i, o = sh(is_eq), sh(oos_eq)
    return round(o/i,3) if i>0 else 0, round(i,3), round(o,3)

def run_exp(label, close_df, sectors, aux):
    gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices = aux
    sig = v11b.precompute(close_df)
    eq, avg_to, rh, bh = v11b.run_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices)
    m = v11b.compute_metrics(eq)
    wfr, is_sh, oos_sh = wf_score(eq)
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + min(m['cagr'],1.0)*0.2
    flag = '✅' if wfr >= 0.60 else '⚠️' if wfr >= 0.55 else '❌'
    print(f"\n{label}")
    print(f"  CAGR={m['cagr']*100:.1f}%  MaxDD={m['max_dd']*100:.1f}%  "
          f"Sh={m['sharpe']:.2f}  Cal={m['calmar']:.2f}  Comp={comp:.3f}  "
          f"WF={wfr}{flag}  IS={is_sh:.2f}→OOS={oos_sh:.2f}  TO={avg_to*100:.1f}%")
    # sector distribution
    sec_cnt = {}
    for t, s in sectors.items():
        sec_cnt[s] = sec_cnt.get(s, 0) + 1
    top_secs = sorted(sec_cnt.items(), key=lambda x: -x[1])[:5]
    print(f"  宇宙={close_df.shape[1]}只  行业分布: " + ", ".join(f"{s}({n})" for s,n in top_secs))
    return m, wfr, is_sh, oos_sh, comp, eq

# ─── main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("🐻 v14 — Nasdaq-100 vs S&P 500 宇宙对比")
    print("=" * 70)

    # 下载缺失的 NDX 股票
    ndx_all = []
    ndx_sectors = {}
    for sec, tks in NDX100.items():
        for t in tks:
            ndx_all.append(t)
            ndx_sectors[t] = sec

    need = [t for t in ndx_all if not (STOCKS / f'{t}.csv').exists()]
    if need:
        print(f"\n[0] 下载 {len(need)} 只 NDX 股票...")
        ok, fail = 0, 0
        for i, t in enumerate(need):
            if stooq_download(t, STOCKS / f'{t}.csv'): ok += 1
            else: fail += 1
            time.sleep(0.3)
            if (i+1) % 20 == 0:
                print(f"  [{i+1}/{len(need)}] ok={ok} fail={fail}")
                time.sleep(2)
        print(f"  Done: ok={ok} fail={fail}")

    # 辅助数据
    gld_p  = lc('GLD'); gdx_p = lc('GDX'); gdxj_p = lc('GDXJ')
    shy_p  = lc('SHY'); tlt_p = lc('TLT'); ief_p  = lc('IEF')
    xlv_p  = lc('XLV'); xlp_p = lc('XLP'); xlu_p  = lc('XLU')
    def_prices = {'XLV': xlv_p, 'XLP': xlp_p, 'XLU': xlu_p}
    aux = (gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices)

    # S&P 500 baseline
    sp500_tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    sp500_sectors = json.load(open(CACHE / "sp500_sectors.json"))
    sp500_close = load_close(sp500_tickers + ['SPY']).loc['2015-01-01':'2025-12-31']

    print("\n[1] S&P 500 Baseline...")
    m_sp, wf_sp, is_sp, oos_sp, comp_sp, eq_sp = run_exp(
        "📊 Baseline — S&P 500", sp500_close, sp500_sectors, aux)

    # NDX-100 全量
    ndx_close = load_close(ndx_all + ['SPY']).loc['2015-01-01':'2025-12-31']
    available_ndx = [t for t in ndx_all if t in ndx_close.columns]
    print(f"\n[2] NDX-100（可用 {len(available_ndx)}/{len(ndx_all)} 只）...")
    m_ndx, wf_ndx, is_ndx, oos_ndx, comp_ndx, eq_ndx = run_exp(
        "📊 NDX-100 Only", ndx_close, ndx_sectors, aux)

    # NDX-100 + S&P 500 混合（用两套 sectors）
    combined_close = load_close(list(set(sp500_tickers + ndx_all)) + ['SPY']).loc['2015-01-01':'2025-12-31']
    combined_secs = {**sp500_sectors, **ndx_sectors}  # NDX 覆盖 SP500 sector（若有冲突）
    print(f"\n[3] NDX-100 + S&P 500 混合...")
    m_mix, wf_mix, is_mix, oos_mix, comp_mix, eq_mix = run_exp(
        "📊 NDX+SP500 Mix", combined_close, combined_secs, aux)

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📋 对比汇总")
    print("=" * 70)
    print(f"{'策略':<18} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} {'WF':>7} {'Comp':>7}")
    print("-" * 60)
    for label, m, wfr, comp in [
        ("S&P 500",     m_sp,  wf_sp,  comp_sp),
        ("NDX-100",     m_ndx, wf_ndx, comp_ndx),
        ("NDX+SP500",   m_mix, wf_mix, comp_mix),
    ]:
        flag = '✅' if wfr >= 0.60 else '⚠️' if wfr >= 0.55 else '❌'
        print(f"  {label:<16} {m['cagr']*100:>6.1f}%  {m['max_dd']*100:>6.1f}%  "
              f"{m['sharpe']:>6.2f}  {wfr:>5.3f}{flag}  {comp:>6.3f}")

    # 保存
    eq_df = pd.DataFrame({'SP500': eq_sp, 'NDX100': eq_ndx, 'NDX_SP500': eq_mix})
    eq_df.to_csv(BASE / 'stocks/codebear/v14_ndx_equity.csv')
    print("\n✓ Equity 已保存 → v14_ndx_equity.csv")
