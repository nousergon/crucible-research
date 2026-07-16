"""
data/fetchers/insider_fetcher.py — SEC EDGAR Form 4 insider trading scanner (O13).

Fetches insider transactions directly from the SEC EDGAR REST API (no edgartools
dependency), detects cluster buying patterns (3+ C-level insiders buying within
30 days), and provides insider sentiment signals for research scoring.

SEC EDGAR rate limit: 10 req/sec — 0.25s delay between requests.
Only fetched for buy candidates (~30-50 tickers), not the full S&P 900.

API endpoints used:
  - Company CIK lookup: efts.sec.gov/LATEST/search-index?q={ticker}&dateRange=custom&...
  - Recent filings: data.sec.gov/submissions/CIK{cik}.json
  - Form 4 XML: sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta

from nousergon_lib.secrets import get_secret
from typing import Optional
from xml.etree import ElementTree

import requests

log = logging.getLogger(__name__)

_RATE_LIMIT_DELAY = 0.25  # stay under 10 req/sec limit
_MAX_FILINGS = 20  # cap Form 4 filings per ticker
_TIMEOUT = 15

# SEC EDGAR requires a User-Agent with company/person name + email.
# Set via EDGAR_IDENTITY env var, format: "Name email@domain.com"
_EDGAR_BASE = "https://data.sec.gov"
_EFTS_BASE = "https://efts.sec.gov/LATEST"

# Process-lifetime cache for the company_tickers.json file (~12 MB). SEC
# rate-limits IPs that fetch this too often; 2026-04-11 saw a 429 on the
# Saturday run. Caching keeps the insider fetcher to one fetch per Lambda
# container lifetime instead of one per invocation per call site.
_COMPANY_TICKERS_CACHE: dict[str, str] | None = None


def _get_headers() -> dict[str, str]:
    """Build SEC-compliant request headers from EDGAR_IDENTITY env var."""
    identity = get_secret("EDGAR_IDENTITY", required=False, default="")
    if not identity:
        raise RuntimeError("EDGAR_IDENTITY env var not set")
    return {
        "User-Agent": identity,
        "Accept": "application/json",
    }


def _fetch_company_tickers(headers: dict) -> dict[str, str]:
    """Fetch + cache the SEC company_tickers.json file as {TICKER: CIK}.

    Returns an empty dict on failure after 3 retries with exponential
    backoff. Shared by _lookup_cik and the batch fetch_insider_activity
    path so we only make one round-trip per Lambda container lifetime.
    """
    global _COMPANY_TICKERS_CACHE
    if _COMPANY_TICKERS_CACHE is not None:
        return _COMPANY_TICKERS_CACHE

    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(tickers_url, headers=headers, timeout=_TIMEOUT)
            # 429 is retryable; raise_for_status() throws HTTPError for it
            resp.raise_for_status()
            data = resp.json()
            cache: dict[str, str] = {}
            for entry in data.values():
                t = entry.get("ticker", "").upper()
                if t:
                    cache[t] = str(entry["cik_str"]).zfill(10)
            _COMPANY_TICKERS_CACHE = cache
            log.debug("Cached %d SEC ticker→CIK entries", len(cache))
            return cache
        except requests.HTTPError as e:
            last_exc = e
            status = e.response.status_code if e.response is not None else None
            if status == 429 and attempt < 3:
                # 2**attempt: 2s, 4s, 8s. Exponential backoff keeps us
                # well clear of SEC's burst window on the next try.
                delay = 2 ** attempt
                log.warning(
                    "SEC company_tickers 429 (attempt %d/3) — backing off %ds",
                    attempt, delay,
                )
                time.sleep(delay)
                continue
            break
        except Exception as e:
            last_exc = e
            break

    log.warning("Failed to fetch SEC company tickers after 3 attempts: %s", last_exc)
    _COMPANY_TICKERS_CACHE = {}
    return _COMPANY_TICKERS_CACHE


def _lookup_cik(ticker: str, headers: dict) -> Optional[str]:
    """Look up CIK number for a ticker symbol via SEC EDGAR company tickers JSON."""
    cache = _fetch_company_tickers(headers)
    return cache.get(ticker.upper())


def _get_form4_filings(
    cik: str, headers: dict, max_filings: int = _MAX_FILINGS
) -> list[dict]:
    """Fetch recent Form 4 filing metadata from SEC EDGAR submissions API."""
    url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=headers, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        filings = []
        for i, form in enumerate(forms):
            if form == "4" and i < len(dates) and i < len(accessions):
                filings.append({
                    "filing_date": dates[i],
                    "accession": accessions[i].replace("-", ""),
                    "accession_dash": accessions[i],
                    "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
                })
                if len(filings) >= max_filings:
                    break

        return filings
    except Exception as e:
        log.debug("Form 4 listing failed for CIK %s: %s", cik, e)
        return []


def _parse_form4_xml(
    cik: str, filing: dict, headers: dict
) -> list[dict]:
    """Download and parse a Form 4 XML filing for transaction details."""
    transactions = []
    accession = filing["accession"]
    primary_doc = filing["primary_doc"]

    # Try the primary document first; fall back to common Form 4 XML name
    urls_to_try = []
    if primary_doc and primary_doc.endswith(".xml"):
        urls_to_try.append(
            f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{primary_doc}"
        )
    urls_to_try.append(
        f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/primary_doc.xml"
    )

    xml_text = None
    for url in urls_to_try:
        try:
            resp = requests.get(url, headers=headers, timeout=_TIMEOUT)
            if resp.status_code == 200 and resp.text.strip().startswith("<"):
                xml_text = resp.text
                break
        except Exception:
            continue

    if not xml_text:
        return transactions

    try:
        # Strip namespace for simpler parsing
        xml_clean = re.sub(r'\sxmlns[^"]*"[^"]*"', "", xml_text)
        root = ElementTree.fromstring(xml_clean)

        # Extract reporting owner info
        owner_elem = root.find(".//reportingOwner")
        owner_name = "Unknown"
        owner_title = ""
        if owner_elem is not None:
            name_elem = owner_elem.find(".//rptOwnerName")
            if name_elem is not None and name_elem.text:
                owner_name = name_elem.text.strip()
            title_elem = owner_elem.find(".//officerTitle")
            if title_elem is not None and title_elem.text:
                owner_title = title_elem.text.strip()

        # Parse non-derivative transactions
        for txn_elem in root.findall(".//nonDerivativeTransaction"):
            try:
                # Transaction code: P=Purchase, S=Sale, A=Grant, etc.
                code_elem = txn_elem.find(".//transactionCoding/transactionCode")
                code = code_elem.text.strip() if code_elem is not None and code_elem.text else ""

                # Only care about open-market purchases (P) and sales (S)
                if code not in ("P", "S"):
                    continue

                shares_elem = txn_elem.find(
                    ".//transactionAmounts/transactionShares/value"
                )
                price_elem = txn_elem.find(
                    ".//transactionAmounts/transactionPricePerShare/value"
                )
                acq_disp_elem = txn_elem.find(
                    ".//transactionAmounts/transactionAcquiredDisposedCode/value"
                )

                shares = float(shares_elem.text) if shares_elem is not None and shares_elem.text else 0
                price = float(price_elem.text) if price_elem is not None and price_elem.text else 0
                acq_disp = acq_disp_elem.text.strip() if acq_disp_elem is not None and acq_disp_elem.text else ""

                txn_type = "BUY" if acq_disp == "A" else "SELL"

                transactions.append({
                    "date": filing["filing_date"],
                    "insider": owner_name,
                    "title": owner_title,
                    "type": txn_type,
                    "shares": int(shares),
                    "value": round(shares * price, 2),
                })
            except Exception as e:
                log.debug("Skipping Form 4 transaction for CIK %s: %s", cik, e)
                continue

    except ElementTree.ParseError as e:
        log.debug("XML parse error for CIK %s accession %s: %s", cik, accession, e)

    return transactions


def fetch_insider_activity(
    tickers: list[str],
    lookback_days: int = 90,
    reference_date: Optional[str] = None,
) -> dict[str, dict]:
    """
    Fetch insider trading activity for a list of tickers from SEC EDGAR.

    Uses the SEC EDGAR REST API directly (no edgartools dependency) to fetch
    Form 4 filings and detect cluster buying patterns.

    Returns per ticker:
        cluster_buy: bool — True if 3+ unique insiders bought in last 30 days
        unique_buyers_30d: int — count of unique insiders who bought in last 30 days
        total_buy_value_30d: float — total dollar value of buys in last 30 days
        net_sentiment: float — net buy/sell ratio (-1 to +1)
        transactions: list[dict] — top 10 recent transactions for display
    """
    today = datetime.strptime(reference_date, "%Y-%m-%d") if reference_date else datetime.now()
    start_date = today - timedelta(days=lookback_days)
    results: dict[str, dict] = {}

    # EDGAR_IDENTITY is required for SEC User-Agent compliance.
    if not get_secret("EDGAR_IDENTITY", required=False):
        log.warning("EDGAR_IDENTITY not set — insider data unavailable. "
                     "Set to 'Name email@domain.com' for SEC EDGAR access.")
        for ticker in tickers:
            results[ticker] = _empty_result()
        return results

    try:
        headers = _get_headers()
    except RuntimeError as e:
        log.warning("SEC EDGAR headers error: %s", e)
        return {t: _empty_result() for t in tickers}

    # Use the shared cache so batch + single-ticker lookup share one round-trip
    cik_map = _fetch_company_tickers(headers)
    if not cik_map:
        return {t: _empty_result() for t in tickers}
    time.sleep(_RATE_LIMIT_DELAY)

    for ticker in tickers:
        try:
            cik = cik_map.get(ticker.upper())
            if not cik:
                log.debug("No CIK found for %s", ticker)
                results[ticker] = _empty_result()
                continue

            filings = _get_form4_filings(cik, headers)
            time.sleep(_RATE_LIMIT_DELAY)

            transactions: list[dict] = []
            for filing in filings:
                filing_date = datetime.strptime(filing["filing_date"], "%Y-%m-%d")
                if filing_date < start_date:
                    continue

                days_ago = (today - filing_date).days
                txns = _parse_form4_xml(cik, filing, headers)
                for txn in txns:
                    txn["days_ago"] = days_ago
                transactions.extend(txns)
                time.sleep(_RATE_LIMIT_DELAY)

            # Cluster detection: unique buyers in last 30 days
            buys_30d = [t for t in transactions if t["type"] == "BUY" and t["days_ago"] <= 30]
            unique_buyers = len(set(t["insider"] for t in buys_30d))
            total_buy_value = sum(t["value"] for t in buys_30d)

            # Net sentiment: ratio of buy value to total value
            total_buys = sum(t["value"] for t in transactions if t["type"] == "BUY")
            total_sells = sum(t["value"] for t in transactions if t["type"] == "SELL")
            total_value = total_buys + total_sells
            if total_value > 0:
                net_sentiment = round((total_buys - total_sells) / total_value, 3)
            else:
                net_sentiment = 0.0

            results[ticker] = {
                "cluster_buy": unique_buyers >= 3,
                "unique_buyers_30d": unique_buyers,
                "total_buy_value_30d": round(total_buy_value, 2),
                "net_sentiment": net_sentiment,
                "transactions": transactions[:10],
            }

        except Exception as e:
            log.warning("Insider data fetch failed for %s: %s", ticker, e)
            results[ticker] = _empty_result()

    log.info("Fetched insider activity for %d/%d tickers", len(results), len(tickers))
    return results


def _empty_result() -> dict:
    """Return neutral insider data when fetching fails."""
    return {
        "cluster_buy": False,
        "unique_buyers_30d": 0,
        "total_buy_value_30d": 0.0,
        "net_sentiment": 0.0,
        "transactions": [],
    }


def format_insider_summary(insider_data: dict) -> str:
    """
    Format insider activity data into a human-readable summary for
    the research agent prompt.

    Returns empty string if no meaningful activity.
    """
    if not insider_data or insider_data.get("unique_buyers_30d", 0) == 0:
        transactions = insider_data.get("transactions", [])
        if not transactions:
            return ""
        # Only sells or no 30d activity
        sells = [t for t in transactions if t["type"] == "SELL"]
        if not sells:
            return ""
        lines = ["Insider Activity (90 days):"]
        lines.append(f"- Net sentiment: {'BEARISH' if insider_data.get('net_sentiment', 0) < -0.3 else 'NEUTRAL'}")
        for t in sells[:3]:
            lines.append(f"  - {t['insider']}: SELL {t['shares']:,} shares (${t['value']:,.0f}) on {t['date']}")
        return "\n".join(lines)

    lines = ["Insider Activity (90 days):"]
    unique_buyers = insider_data["unique_buyers_30d"]
    cluster = insider_data.get("cluster_buy", False)

    if cluster:
        lines.append(f"- {unique_buyers} unique insiders bought in last 30 days (CLUSTER BUYING detected)")
    elif unique_buyers > 0:
        lines.append(f"- {unique_buyers} insider(s) bought in last 30 days")

    # Show top transactions
    for t in insider_data.get("transactions", [])[:5]:
        action = t["type"]
        lines.append(
            f"  - {t['insider']} {action} {t['shares']:,} shares "
            f"(${t['value']:,.0f}) on {t['date']}"
        )

    sentiment = insider_data.get("net_sentiment", 0)
    if sentiment > 0.3:
        lines.append("- Net sentiment: BULLISH")
    elif sentiment < -0.3:
        lines.append("- Net sentiment: BEARISH")
    else:
        lines.append("- Net sentiment: NEUTRAL")

    return "\n".join(lines)


def cache_insider_to_s3(
    data: dict[str, dict],
    date_str: str,
    bucket: str = "alpha-engine-research",
) -> None:
    """Cache insider data to S3."""
    try:
        import boto3
        s3 = boto3.client("s3")
        key = f"archive/insider/{date_str}.json"
        # Strip non-serializable transaction details for cache
        cache_data = {}
        for ticker, d in data.items():
            cache_data[ticker] = {
                k: v for k, v in d.items() if k != "transactions"
            }
            cache_data[ticker]["n_transactions"] = len(d.get("transactions", []))
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(cache_data, default=str),
            ContentType="application/json",
        )
        log.info("Cached insider data to s3://%s/%s", bucket, key)
    except Exception as e:
        log.warning("Failed to cache insider data to S3: %s", e)
