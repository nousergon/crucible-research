"""
Scanner timing validation script.

Measures how long the Stage 1 quant filter takes for increasing universe sizes.
Run this locally before deploying to Lambda to determine a safe cap.

Usage:
    python local/time_scanner.py

Outputs timing for 150, 300, 500, and full universe sizes.
If 500 tickers completes in < 120s, the full ~900 should be safe for Lambda (600s timeout).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

# Imported after load_dotenv() above so a local .env override of
# config-time env vars (e.g. S3_BUCKET/AWS_REGION) takes effect.
import yfinance as yf  # noqa: E402

from config import UNIVERSE_TICKERS  # noqa: E402
from data.fetchers.price_fetcher import compute_technical_indicators, fetch_sp500_sp400_tickers  # noqa: E402
from scoring.technical import compute_technical_score  # noqa: E402


def time_download(tickers: list[str], label: str) -> float:
    print(f"\n{label}: {len(tickers)} tickers", flush=True)
    t0 = time.time()
    try:
        df = yf.download(
            tickers=tickers,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        elapsed = time.time() - t0
        print(f"  Download:  {elapsed:.1f}s", flush=True)

        # Compute technical indicators for all tickers
        t1 = time.time()
        scored = 0
        for ticker in tickers:
            try:
                tick_df = df[ticker] if len(tickers) > 1 else df
                tick_df = tick_df.dropna(subset=["Close"])
                indicators = compute_technical_indicators(tick_df)
                if indicators:
                    compute_technical_score(indicators, market_regime="neutral")
                    scored += 1
            except Exception as e:
                print(f"  skip {ticker}: {e}", flush=True)
        t2 = time.time()
        print(f"  Scoring:   {t2 - t1:.1f}s ({scored} tickers scored)", flush=True)
        print(f"  Total:     {t2 - t0:.1f}s", flush=True)
        return t2 - t0
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ERROR after {elapsed:.1f}s: {e}", flush=True)
        return elapsed


def main():
    print("Fetching universe list...")
    all_tickers = fetch_sp500_sp400_tickers()
    exclude = set(UNIVERSE_TICKERS)
    scanner_universe = [t for t in all_tickers if t not in exclude]
    print(f"Full scanner universe: {len(scanner_universe)} tickers")

    results = {}

    for size, label in [(150, "Sample A"), (300, "Sample B"), (500, "Sample C"),
                         (len(scanner_universe), "Full universe")]:
        sample = scanner_universe[:size]
        elapsed = time_download(sample, label)
        results[size] = elapsed

        if elapsed > 180:
            print(f"\n⚠  {size} tickers took {elapsed:.0f}s — approaching Lambda limits.")
            print("  Recommend keeping scanner_universe_sample cap at the previous size.")
            break

    print("\n=== SUMMARY ===")
    for size, elapsed in results.items():
        safety = "✓ safe" if elapsed < 120 else ("⚠ borderline" if elapsed < 240 else "✗ too slow")
        print(f"  {size:4d} tickers: {elapsed:5.1f}s  {safety}")

    print()
    if results:
        last_size = max(results)
        last_time = results[last_size]
        if last_time < 120:
            print(f"Recommendation: full universe ({last_size}) is fine. Remove the 150-ticker cap in graph/research_graph.py.")
            print("  In graph/research_graph.py, change:")
            print("    scanner_universe_sample = scanner_universe[:150]")
            print("  to:")
            print("    scanner_universe_sample = scanner_universe")
        else:
            safe_sizes = [s for s, t in results.items() if t < 120]
            cap = max(safe_sizes) if safe_sizes else 150
            print(f"Recommendation: cap scanner universe at {cap} tickers.")
            print("  In graph/research_graph.py, update:")
            print(f"    scanner_universe_sample = scanner_universe[:{cap}]")


if __name__ == "__main__":
    main()
