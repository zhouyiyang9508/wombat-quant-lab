"""
Momentum v13 — 中小盘扩展实验
================================
在 v11b 基础上，扩大股票宇宙：
  - 原版：S&P 500（502只）
  - 新版：S&P 500 + S&P 400中盘 + 部分S&P 600小盘（合计约700-800只）

研究问题：中小盘动量因子能否提升 CAGR？代价是什么？

代码熊 🐻 | 2026-02-22
"""
import pandas as pd, numpy as np, json, sys, time, urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / 'data_cache'
STOCKS = CACHE / 'stocks'
sys.path.insert(0, str(BASE))
import stocks.codebear.momentum_v11b_final as v11b

# ─── 中小盘候选 ticker 列表（按 GICS 行业分组）─────────────────────────────
# 主要来自 S&P 400 中盘 + 部分 S&P 600 小盘（2024年已有稳定交易历史的股票）
MIDSMALL_TICKERS = {
    # Technology
    'Technology': [
        'GDDY', 'FFIV', 'JNPR', 'WEX', 'EVTC', 'SPSC', 'CSGS', 'BLKB',
        'NTCT', 'CVLT', 'PLUS', 'PRFT', 'SAIC', 'CACI', 'LDOS', 'SAIC',
        'SCI', 'EXLS', 'EPAM', 'VRNS', 'ALTR', 'DIGI', 'NATI', 'TTEC',
    ],
    # Healthcare
    'Healthcare': [
        'PRGO', 'GMED', 'PDCO', 'MMSI', 'OMCL', 'LMAT', 'USPH', 'SEM',
        'ACAD', 'HALO', 'ITGR', 'SUPN', 'INVA', 'PCRX', 'ADUS', 'AMSF',
        'AFYA', 'NTRA', 'OMAB', 'PGNY', 'HIMS', 'QDEL', 'TMDX',
    ],
    # Industrials
    'Industrials': [
        'TREX', 'UFPI', 'NPO', 'LBRT', 'WLDN', 'ENS', 'BWXT', 'MTZ',
        'MWA', 'ROAD', 'GATX', 'SXC', 'CLH', 'ACCO', 'HURN', 'VSE',
        'HSII', 'KFRC', 'RUSHA', 'GMS', 'IBP', 'CTOS', 'DRVN', 'ASTL',
    ],
    # ConsumerDisc
    'ConsumerDisc': [
        'WEN', 'JACK', 'BJRI', 'CBRL', 'CAKE', 'EAT', 'DINE',
        'ARKO', 'PLAY', 'LOCO', 'SHAK', 'NAPA', 'PTRY',
        'OLLI', 'BJ', 'BURL', 'RH', 'PRPL', 'LESL', 'WSM',
        'M', 'URBN', 'ANF', 'BOOT', 'GOOS', 'ONON', 'BROS',
    ],
    # Financials
    'Financials': [
        'SNV', 'BOH', 'WSFS', 'CADE', 'UMBF', 'FNB', 'HOPE', 'CVBF',
        'TCBI', 'GABC', 'FFIN', 'HTLF', 'IBOC', 'BANF', 'TRMK',
        'PNFP', 'SFNC', 'OFG', 'HFWA', 'CATY', 'BYFC', 'SSBK',
        'SFBS', 'TOWN', 'FBIZ', 'PFIS',
    ],
    # Energy
    'Energy': [
        'CIVI', 'MGY', 'SBOW', 'PUMP', 'NR', 'OMP', 'VVV',
        'TRGP', 'WTTR', 'DKL', 'HESS', 'PAA', 'CAPL',
    ],
    # Materials
    'Materials': [
        'OLN', 'SXT', 'TROX', 'HWKN', 'BCPC', 'CBT', 'NEU',
        'ASIX', 'IOSP', 'CRS', 'SMP', 'KALU', 'ZEUS', 'CMC', 'STLD',
    ],
    # RealEstate
    'RealEstate': [
        'NNN', 'NSA', 'IRT', 'PGRE', 'GNL', 'ILPT', 'NXRT',
        'IIPR', 'VRE', 'CLPR', 'AIV', 'BRT', 'AFIN',
    ],
    # Utilities
    'Utilities': [
        'SJW', 'AWR', 'MSEX', 'ARTNA', 'SWX', 'YORW', 'CTWS',
        'CWCO', 'NWPX', 'MGEE', 'NSPM',
    ],
    # ConsumerStap
    'ConsumerStap': [
        'UTZ', 'SMPL', 'LANC', 'BGS', 'CHEF', 'COKE', 'SENEA',
        'JBSS', 'CALM', 'NTRI', 'HAIN', 'TWNK', 'MGPI',
    ],
    # CommServices
    'CommServices': [
        'SATS', 'TDS', 'USM', 'IRDM', 'LUMN', 'IPAR',
        'RDI', 'IHRT', 'AMCX', 'NWSA', 'NWSL',
    ],
}

def stooq_download(ticker, out_path):
    """从 Stooq 下载一只股票数据"""
    sym = ticker.lower().replace('-', '.') + '.us'
    url = f'https://stooq.com/q/d/l/?s={sym}&d1=20140601&d2=20251231&i=d'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0)'})
        data = urllib.request.urlopen(req, timeout=15).read().decode()
        lines = data.strip().split('\n')
        if len(lines) > 200 and 'Date' in lines[0]:
            out_path.write_text(data)
            return True
        return False
    except Exception:
        return False

def download_midsmall(force=False):
    """下载中小盘股票数据（已有的跳过）"""
    all_tickers = []
    for sec, tks in MIDSMALL_TICKERS.items():
        all_tickers.extend(tks)
    
    need = [t for t in set(all_tickers) if force or not (STOCKS / f'{t}.csv').exists()]
    if not need:
        print(f"  All {len(set(all_tickers))} mid/small tickers already cached.")
        return
    
    print(f"  Downloading {len(need)} mid/small cap tickers from Stooq...")
    ok, fail = 0, 0
    for i, t in enumerate(need):
        if stooq_download(t, STOCKS / f'{t}.csv'):
            ok += 1
        else:
            fail += 1
        if (i+1) % 20 == 0:
            print(f"    [{i+1}/{len(need)}] ok={ok} fail={fail}")
            time.sleep(3)
        else:
            time.sleep(0.3)
    print(f"  Done: ok={ok}, fail={fail}, total={len(need)}")

def load_close(tickers):
    """加载股票收盘价，返回 DataFrame"""
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
        if len(s) < 500: continue  # 至少2年数据
        frames[t] = s
    if not frames: return pd.DataFrame()
    return pd.DataFrame(frames)

def wf_score(eq, is_end='2021-12-31', oos_start='2022-01-01', rf=0.04):
    is_eq  = eq[eq.index <= pd.Timestamp(is_end)]
    oos_eq = eq[eq.index >= pd.Timestamp(oos_start)]
    if len(is_eq) < 12 or len(oos_eq) < 12: return 0, 0, 0
    def sh(e):
        r = e.pct_change().dropna()
        return float((r.mean()-rf/12)/r.std()*np.sqrt(12)) if r.std()>0 else 0
    i, o = sh(is_eq), sh(oos_eq)
    return round(o/i,3) if i>0 else 0, round(i,3), round(o,3)

def run_experiment(label, close_df, sectors, sp500_sectors, aux):
    """运行单次回测，返回指标"""
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
    return m, wfr, is_sh, oos_sh, comp, eq

# ─── main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("🐻 v13 — 中小盘扩展实验")
    print("=" * 70)
    
    # 1. 下载中小盘数据
    print("\n[1/4] 下载中小盘股票数据...")
    download_midsmall()
    
    # 2. 构建合并宇宙 + sectors
    print("\n[2/4] 构建股票宇宙...")
    
    # 原 S&P 500
    sp500_tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    sp500_sectors = json.load(open(CACHE / "sp500_sectors.json"))
    
    # 中小盘
    midsmall_sectors = {}
    for sec, tks in MIDSMALL_TICKERS.items():
        for t in tks:
            midsmall_sectors[t] = sec
    
    # 过滤实际有数据的中小盘 ticker
    available_ms = [t for t in midsmall_sectors if (STOCKS / f'{t}.csv').exists()]
    print(f"  S&P500 tickers: {len(sp500_tickers)}")
    print(f"  Mid/Small tickers with data: {len(available_ms)} / {len(midsmall_sectors)}")
    
    # 加载辅助数据
    def lc(name):
        fp = CACHE / f'{name}.csv'
        if not fp.exists(): return pd.Series(dtype=float)
        df = pd.read_csv(fp)
        c = 'Date' if 'Date' in df.columns else df.columns[0]
        df[c] = pd.to_datetime(df[c])
        return df.set_index(c).sort_index()['Close'].dropna()
    
    gld_p  = lc('GLD'); gdx_p  = lc('GDX'); gdxj_p = lc('GDXJ')
    shy_p  = lc('SHY'); tlt_p  = lc('TLT'); ief_p  = lc('IEF')
    xlv_p  = lc('XLV'); xlp_p  = lc('XLP'); xlu_p  = lc('XLU')
    def_prices = {'XLV': xlv_p, 'XLP': xlp_p, 'XLU': xlu_p}
    aux = (gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices)
    
    # 3. 实验 A — Baseline S&P 500
    print("\n[3/4] 运行对照实验...")
    
    sp500_close = load_close(sp500_tickers + ['SPY'])
    sp500_close = sp500_close.loc['2015-01-01':'2025-12-31']
    print(f"  Baseline 宇宙: {sp500_close.shape[1]} 只股票")
    
    m_base, wf_base, is_base, oos_base, comp_base, eq_base = run_experiment(
        "📊 Baseline — S&P 500 only", sp500_close, sp500_sectors, sp500_sectors, aux)
    
    # 4. 实验 B — S&P 500 + 中小盘
    print("\n[4/4] 运行中小盘扩展实验...")
    
    combined_tickers = sp500_tickers + available_ms + ['SPY']
    combined_sectors = {**sp500_sectors, **midsmall_sectors}
    
    combined_close = load_close(combined_tickers)
    combined_close = combined_close.loc['2015-01-01':'2025-12-31']
    print(f"  扩展宇宙: {combined_close.shape[1]} 只股票")
    
    m_exp, wf_exp, is_exp, oos_exp, comp_exp, eq_exp = run_experiment(
        "📊 Extended — S&P 500 + Mid/Small Cap", combined_close, combined_sectors, combined_sectors, aux)
    
    # 汇总对比
    print("\n" + "=" * 70)
    print("📋 对比结果")
    print("=" * 70)
    delta_cagr = (m_exp['cagr'] - m_base['cagr']) * 100
    delta_dd   = (m_exp['max_dd'] - m_base['max_dd']) * 100
    delta_sh   = m_exp['sharpe'] - m_base['sharpe']
    delta_wf   = wf_exp - wf_base
    delta_comp = comp_exp - comp_base
    
    print(f"  CAGR:    {m_base['cagr']*100:.1f}% → {m_exp['cagr']*100:.1f}% ({delta_cagr:+.1f}%)")
    print(f"  MaxDD:   {m_base['max_dd']*100:.1f}% → {m_exp['max_dd']*100:.1f}% ({delta_dd:+.1f}%)")
    print(f"  Sharpe:  {m_base['sharpe']:.2f} → {m_exp['sharpe']:.2f} ({delta_sh:+.2f})")
    print(f"  WF:      {wf_base:.3f} → {wf_exp:.3f} ({delta_wf:+.3f})")
    print(f"  Comp:    {comp_base:.3f} → {comp_exp:.3f} ({delta_comp:+.3f})")
    
    # 保存 equity
    eq_df = pd.DataFrame({'SP500_only': eq_base, 'WithMidSmall': eq_exp})
    eq_df.to_csv(BASE / 'stocks/codebear/v13_midsmall_equity.csv')
    
    verdict = "✅ 中小盘有效" if delta_cagr > 1.0 and wf_exp >= 0.60 else \
              "⚠️ 收益提升但 WF 下降" if delta_cagr > 1.0 else \
              "❌ 中小盘无效"
    print(f"\n{verdict}")
