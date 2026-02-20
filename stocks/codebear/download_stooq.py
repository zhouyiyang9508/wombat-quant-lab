#!/usr/bin/env python3
"""Download S&P 500 stock data from Stooq."""
import urllib.request, time, os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache" / "stocks"
CACHE.mkdir(parents=True, exist_ok=True)

tickers = (BASE / "data_cache" / "sp500_tickers.txt").read_text().strip().split('\n')
tickers.append('SPY')

def download_one(ticker):
    f = CACHE / f"{ticker}.csv"
    if f.exists() and f.stat().st_size > 1000:
        return True
    
    # Stooq uses lowercase with .us suffix for US stocks
    stooq_sym = ticker.lower().replace('-', '.') + '.us'
    url = f'https://stooq.com/q/d/l/?s={stooq_sym}&d1=20140601&d2=20251231&i=d'
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        data = urllib.request.urlopen(req, timeout=15).read().decode()
        lines = data.strip().split('\n')
        if len(lines) > 100 and 'Date' in lines[0]:
            f.write_text(data)
            return True
        return False
    except Exception as e:
        return False

total = len(tickers)
success = 0
fail = 0
skip = 0

for i, t in enumerate(tickers):
    f = CACHE / f"{t}.csv"
    if f.exists() and f.stat().st_size > 1000:
        skip += 1
        continue
    
    ok = download_one(t)
    if ok:
        success += 1
    else:
        fail += 1
    
    if (i + 1) % 20 == 0:
        print(f"Progress: {i+1}/{total} (ok={success}, fail={fail}, skip={skip})")
    
    time.sleep(0.3)  # Be polite
    if (i + 1) % 100 == 0:
        print(f"  Pausing 10s...")
        time.sleep(10)

print(f"\nDone! Success={success}, Failed={fail}, Skipped={skip}, Total={total}")
