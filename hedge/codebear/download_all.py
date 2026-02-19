import yfinance as yf
import time, os

DATA_DIR = '/root/.openclaw/workspace/wombat-quant-lab/data_cache'
NEED = ['IWM','EFA','EEM','IEF','HYG','TIP','SLV','USO','DBC','VNQ','SHY']

for t in NEED:
    cache = os.path.join(DATA_DIR, f'{t}.csv')
    if os.path.exists(cache):
        print(f"{t}: cached")
        continue
    print(f"Downloading {t}...")
    time.sleep(8)
    try:
        df = yf.download(t, start='2012-01-01', end='2026-02-19', auto_adjust=True, progress=False)
        if len(df) > 100:
            df.to_csv(cache)
            print(f"  {t}: {len(df)} rows")
        else:
            print(f"  {t}: only {len(df)} rows, skipped")
    except Exception as e:
        print(f"  {t}: FAILED - {e}")
