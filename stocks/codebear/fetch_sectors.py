#!/usr/bin/env python3
"""Fetch GICS sector data for S&P 500 stocks and cache."""
import json, os, time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
SECTOR_FILE = CACHE / "sp500_sectors.json"

def fetch_sectors():
    if SECTOR_FILE.exists():
        with open(SECTOR_FILE) as f:
            sectors = json.load(f)
        print(f"Loaded {len(sectors)} sectors from cache")
        return sectors
    
    import yfinance as yf
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    sectors = {}
    
    for i, t in enumerate(tickers):
        try:
            info = yf.Ticker(t).info
            sector = info.get('sector', 'Unknown')
            sectors[t] = sector
            if (i+1) % 50 == 0:
                print(f"  [{i+1}/{len(tickers)}] {t} → {sector}")
        except Exception as e:
            sectors[t] = 'Unknown'
        time.sleep(0.1)
    
    with open(SECTOR_FILE, 'w') as f:
        json.dump(sectors, f, indent=2)
    print(f"Saved {len(sectors)} sectors to {SECTOR_FILE}")
    return sectors

if __name__ == '__main__':
    s = fetch_sectors()
    from collections import Counter
    c = Counter(s.values())
    for sector, count in c.most_common():
        print(f"  {sector}: {count}")
