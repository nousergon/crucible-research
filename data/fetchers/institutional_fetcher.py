"""
data/fetchers/institutional_fetcher.py — DEPRECATED 2026-07-13 (archived).

Replaced by ``alpha-engine-data/data/derived/inst_ownership.py``, which
builds a per-ticker institutional-ownership snapshot from SEC quarterly
bulk Form 13F data (authoritative, free, no vendor dependency).

Why deprecated:
- ``Company(ticker).get_filings(form="13F-HR")`` queries by the STOCK's
  CIK, but 13F-HR is filed by the FUND MANAGER's CIK — always empty for
  normal operating companies (alpha-engine-config-I2428).
- The ``inst_ownership`` derived table resolves CUSIP→ticker from SEC
  bulk INFOTABLE data, producing real QoQ share/value deltas.

All consumers should use ``data.substrate.reader.read_inst_ownership()``
instead. This module kept as a stub for backward compatibility.
"""

from __future__ import annotations

import logging
import time

from nousergon_lib.secrets import get_secret

logger = logging.getLogger(__name__)

_TICKER_DELAY = 0.25  # delay between ticker lookups (edgartools has built-in 9 req/sec)


def fetch_institutional_accumulation(
    tickers: list[str],
    min_funds_for_signal: int | None = None,
) -> dict[str, dict]:
    """
    Detect institutional accumulation from 13F-HR filings.

    For each ticker, finds top institutional holders' 13F-HR filings from
    current + prior quarter and detects position increases.

    Args:
        tickers: List of ticker symbols to analyze
        min_funds_for_signal: Minimum funds accumulating for accumulation_signal=True

    Returns per ticker:
        n_funds_accumulating: int — funds that increased positions QoQ
        total_new_shares: int — total shares added across accumulating funds
        accumulation_signal: bool — True if n_funds_accumulating >= min_funds_for_signal
    """
    # Read min_funds from config if not explicitly provided
    if min_funds_for_signal is None:
        try:
            from config import get_research_params
            min_funds_for_signal = get_research_params()["institutional_min_funds"]
        except Exception:
            min_funds_for_signal = 3

    # Check EDGAR_IDENTITY
    identity = get_secret("EDGAR_IDENTITY", required=False, default="")
    if not identity:
        logger.warning("EDGAR_IDENTITY not set — skipping 13F institutional data")
        return {t: _empty_result() for t in tickers}

    try:
        from edgar import Company, set_identity
        set_identity(identity)
    except ImportError:
        logger.warning("edgartools not installed — skipping 13F institutional data")
        return {t: _empty_result() for t in tickers}
    except Exception as e:
        logger.warning("edgartools identity setup failed: %s", e)
        return {t: _empty_result() for t in tickers}

    results: dict[str, dict] = {}

    for ticker in tickers:
        try:
            result = _analyze_ticker_13f(ticker, Company, min_funds_for_signal)
            results[ticker] = result
        except Exception as e:
            logger.debug("13F analysis failed for %s: %s", ticker, e)
            results[ticker] = _empty_result()

        time.sleep(_TICKER_DELAY)

    n_signals = sum(1 for v in results.values() if v.get("accumulation_signal"))
    logger.info(
        "[13F] analyzed %d tickers, %d with accumulation signal",
        len(results), n_signals,
    )
    return results


def _analyze_ticker_13f(
    ticker: str,
    Company,
    min_funds_for_signal: int,
) -> dict:
    """Analyze 13F filings for a single ticker."""
    company = Company(ticker)

    # Get recent 13F-HR filings (institutional holders)
    filings = company.get_filings(form="13F-HR").latest(5)

    if not filings or len(filings) == 0:
        return _empty_result()

    # Try to get the most recent 13F and compare with the previous one
    n_accumulating = 0
    total_new_shares = 0

    try:
        latest_filing = filings[0]
        thirteen_f = latest_filing.obj()

        if hasattr(thirteen_f, 'previous_holding_report'):
            prev = thirteen_f.previous_holding_report()
            if prev is not None:
                # Compare holdings between current and previous quarter
                current_holdings = {
                    h.cusip: h.value for h in thirteen_f.holdings
                } if hasattr(thirteen_f, 'holdings') else {}

                prev_holdings = {
                    h.cusip: h.value for h in prev.holdings
                } if hasattr(prev, 'holdings') else {}

                for cusip, current_value in current_holdings.items():
                    prev_value = prev_holdings.get(cusip, 0)
                    if current_value and prev_value and current_value > prev_value:
                        n_accumulating += 1
                        total_new_shares += int(current_value - prev_value)
    except Exception as e:
        logger.debug("13F comparison failed for %s: %s", ticker, e)
        # Fall back to just checking if we found institutional filings
        return _empty_result()

    return {
        "n_funds_accumulating": n_accumulating,
        "total_new_shares": total_new_shares,
        "accumulation_signal": n_accumulating >= min_funds_for_signal,
    }


def _empty_result() -> dict:
    """Return neutral institutional data when fetching fails."""
    return {
        "n_funds_accumulating": 0,
        "total_new_shares": 0,
        "accumulation_signal": False,
    }
