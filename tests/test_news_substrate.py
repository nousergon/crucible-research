"""Tests for the news-source substrate — Wave 1 PR A of the institutional
data revamp (`~/Development/alpha-engine-docs/private/data-revamp-260513.md`).

Covers:
  - Pydantic ``NewsArticle`` shape (validation + frozen + extra='forbid')
  - Protocol structural-subtyping for NewsSource adapters
  - Polygon adapter normalization (happy + schema-drift + transient failure)
  - GDELT adapter normalization (happy + query-building + schema-drift)
  - Yahoo RSS adapter wrapping the legacy fetcher
  - Paid-vendor stubs (Benzinga / RavenPack / Bloomberg) raise on init
  - Aggregator fan-in + dedup + trust weighting + symmetric ordering
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from data.aggregator import (
    AggregatedNewsArticle,
    DEFAULT_TRUST_WEIGHTS,
    NewsAggregator,
    _article_fingerprint,
    _normalize_title,
    _url_fingerprint,
)
from data.sources.protocols import NewsArticle, NewsSource
from data.sources.news.benzinga import BenzingaNewsAdapter
from data.sources.news.bloomberg import BloombergNewsAdapter
from data.sources.news.gdelt import GdeltNewsAdapter, _build_query
from data.sources.news.polygon import PolygonNewsAdapter
from data.sources.news.ravenpack import RavenpackNewsAdapter
from data.sources.news.yahoo_rss import YahooRssNewsAdapter


# ── NewsArticle shape ──────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_article(
    *,
    source: str = "polygon",
    title: str = "Earnings beat",
    url: str = "https://example.com/x",
    tickers: tuple[str, ...] = ("AAPL",),
    published_at: datetime | None = None,
) -> NewsArticle:
    return NewsArticle(
        tickers=tickers,
        title=title,
        body_excerpt="lead paragraph",
        url=url,
        published_at=published_at or _now(),
        source=source,
        vendor_article_id="vid-1",
        fetched_at=_now(),
    )


class TestNewsArticleShape:
    def test_canonical_construction(self):
        a = _make_article()
        assert a.source == "polygon"
        assert a.tickers == ("AAPL",)
        # frozen=True — assignment raises
        with pytest.raises(ValidationError):
            a.title = "different"  # type: ignore[misc]

    def test_extra_fields_forbidden(self):
        """Schema drift via extra fields surfaces as ValidationError —
        forces adapters to map vendor-specific fields explicitly."""
        with pytest.raises(ValidationError, match="Extra inputs are not"):
            NewsArticle(
                tickers=("AAPL",),
                title="t",
                body_excerpt="e",
                url="https://x",
                published_at=_now(),
                source="polygon",
                fetched_at=_now(),
                some_unknown_vendor_field="oops",  # type: ignore[call-arg]
            )

    def test_multi_ticker_record(self):
        a = _make_article(tickers=("AAPL", "MSFT", "GOOGL"))
        assert a.tickers == ("AAPL", "MSFT", "GOOGL")


# ── Protocol structural subtyping ──────────────────────────────────────


class TestNewsSourceProtocol:
    def test_polygon_adapter_satisfies_protocol(self):
        adapter = PolygonNewsAdapter(client=MagicMock())
        assert isinstance(adapter, NewsSource)

    def test_gdelt_adapter_satisfies_protocol(self):
        adapter = GdeltNewsAdapter(ticker_name_map={})
        assert isinstance(adapter, NewsSource)

    def test_yahoo_adapter_satisfies_protocol(self):
        adapter = YahooRssNewsAdapter(fetcher=lambda *a, **k: [])
        assert isinstance(adapter, NewsSource)


# ── Polygon adapter ────────────────────────────────────────────────────


class TestPolygonNewsAdapter:
    def test_happy_path_normalizes_to_news_article(self):
        fake_client = MagicMock()
        fake_client._get.return_value = {
            "results": [
                {
                    "id": "abc-123",
                    "title": "NVDA Earnings Beat",
                    "description": "Strong Q4 results",
                    "article_url": "https://example.com/nvda-q4",
                    "published_utc": "2026-05-12T14:30:00Z",
                    "tickers": ["NVDA"],
                    "keywords": ["earnings", "AI"],
                    "author": "Reporter",
                },
            ]
        }
        adapter = PolygonNewsAdapter(client=fake_client)
        out = adapter.fetch(["NVDA"], hours=24)
        assert len(out) == 1
        article = out[0]
        assert article.source == "polygon"
        assert article.title == "NVDA Earnings Beat"
        assert article.vendor_article_id == "abc-123"
        assert article.tickers == ("NVDA",)
        assert "earnings" in article.tags

    def test_transient_failure_returns_partial_batch(self):
        """One ticker fails → keep going for the rest. Defense-in-depth
        per the Protocol contract."""
        fake_client = MagicMock()

        def side_effect(path, params):
            if params["ticker"] == "BROKEN":
                raise RuntimeError("polygon 500")
            return {"results": [{
                "id": f"id-{params['ticker']}",
                "title": "headline",
                "article_url": f"https://x.com/{params['ticker']}",
                "published_utc": "2026-05-12T14:30:00Z",
                "tickers": [params["ticker"]],
            }]}

        fake_client._get.side_effect = side_effect
        adapter = PolygonNewsAdapter(client=fake_client)
        out = adapter.fetch(["AAPL", "BROKEN", "MSFT"], hours=24)
        # AAPL + MSFT come through; BROKEN is skipped
        assert {a.title for a in out} == {"headline"}
        assert {a.url for a in out} == {"https://x.com/AAPL", "https://x.com/MSFT"}

    def test_schema_drift_on_one_item_skips_just_that_item(self):
        """Vendor adds/removes a field → the adapter logs + skips that
        article, doesn't crash the batch."""
        fake_client = MagicMock()
        fake_client._get.return_value = {
            "results": [
                {  # missing required published_utc
                    "id": "incomplete",
                    "title": "no date",
                    "article_url": "https://x.com/1",
                },
                {  # well-formed
                    "id": "good",
                    "title": "good headline",
                    "article_url": "https://x.com/2",
                    "published_utc": "2026-05-12T14:30:00Z",
                    "tickers": ["AAPL"],
                },
            ]
        }
        adapter = PolygonNewsAdapter(client=fake_client)
        out = adapter.fetch(["AAPL"], hours=24)
        assert len(out) == 1
        assert out[0].vendor_article_id == "good"


# ── GDELT adapter ──────────────────────────────────────────────────────


class TestGdeltNewsAdapter:
    def test_happy_path(self):
        fake_http = MagicMock()
        fake_http.get.return_value = MagicMock(
            json=MagicMock(return_value={
                "articles": [{
                    "url": "https://reuters.com/x",
                    "title": "AAPL stock surges",
                    "seendate": "20260512T143000Z",
                    "sourcecountry": "US",
                    "domain": "reuters.com",
                    "language": "English",
                }],
            }),
            raise_for_status=MagicMock(return_value=None),
        )
        adapter = GdeltNewsAdapter(
            ticker_name_map={"AAPL": "Apple Inc"},
            http=fake_http,
            inter_request_sleep=0.0,
        )
        out = adapter.fetch(["AAPL"], hours=24)
        assert len(out) == 1
        assert out[0].source == "gdelt"
        assert out[0].url == "https://reuters.com/x"
        assert out[0].published_at.year == 2026
        # Domain tag preserved
        assert "reuters.com" in out[0].tags

    def test_query_includes_ticker_and_company_name(self):
        q = _build_query("AAPL", "Apple Inc")
        assert "AAPL" in q
        assert '"Apple Inc"' in q  # multi-word names quoted
        assert "sourcecountry:US" in q

    def test_query_handles_single_word_company_name(self):
        q = _build_query("NVDA", "Nvidia")
        assert "Nvidia" in q
        assert '"Nvidia"' not in q  # single-word doesn't quote

    def test_failure_skips_ticker_continues_batch(self):
        fake_http = MagicMock()
        call_count = {"i": 0}

        def get(url, params, timeout):
            call_count["i"] += 1
            if call_count["i"] == 1:
                raise RuntimeError("gdelt rate-limit")
            return MagicMock(
                json=MagicMock(return_value={"articles": [{
                    "url": "https://x.com/y",
                    "title": "ok",
                    "seendate": "20260512T143000Z",
                }]}),
                raise_for_status=MagicMock(return_value=None),
            )

        fake_http.get.side_effect = get
        adapter = GdeltNewsAdapter(
            ticker_name_map={"AAPL": "Apple", "MSFT": "Microsoft"},
            http=fake_http,
            inter_request_sleep=0.0,
        )
        out = adapter.fetch(["AAPL", "MSFT"])
        assert len(out) == 1
        assert out[0].url == "https://x.com/y"

    def test_ticker_falls_back_to_symbol_when_no_name_in_map(self):
        """If the ticker→name map doesn't cover a ticker, the adapter
        uses the ticker itself as the query term (don't crash, don't
        skip)."""
        fake_http = MagicMock()
        fake_http.get.return_value = MagicMock(
            json=MagicMock(return_value={"articles": []}),
            raise_for_status=MagicMock(return_value=None),
        )
        adapter = GdeltNewsAdapter(
            ticker_name_map={},
            http=fake_http,
            inter_request_sleep=0.0,
        )
        # Should not raise
        adapter.fetch(["UNKNOWN"])
        params_used = fake_http.get.call_args.kwargs["params"]
        assert "UNKNOWN" in params_used["query"]


# ── Yahoo RSS adapter ──────────────────────────────────────────────────


class TestYahooRssNewsAdapter:
    def test_wraps_legacy_fetcher(self):
        legacy_output = [
            {
                "headline": "AAPL hits new high",
                "source": "Reuters",
                "url": "https://reuters.com/aapl",
                "published_utc": "2026-05-12T14:30:00Z",
                "article_excerpt": "Apple shares...",
                "article_hash": "h1",
            },
        ]
        fake_fetcher = MagicMock(return_value=legacy_output)
        adapter = YahooRssNewsAdapter(fetcher=fake_fetcher)
        out = adapter.fetch(["AAPL"], hours=24)
        assert len(out) == 1
        assert out[0].source == "yahoo_rss"
        assert out[0].vendor_article_id == "h1"
        fake_fetcher.assert_called_once_with("AAPL", hours=24)

    def test_skips_entries_without_url(self):
        legacy_output = [{"headline": "no url", "published_utc": "2026-05-12T14:30:00Z"}]
        adapter = YahooRssNewsAdapter(fetcher=MagicMock(return_value=legacy_output))
        out = adapter.fetch(["AAPL"])
        assert out == []


# ── Paid-vendor stubs ──────────────────────────────────────────────────


class TestPaidStubsFailLoudOnConstruction:
    """Phase 4 stubs must raise on init, NOT silently return empty —
    a future operator wiring up these adapters needs the loud error to
    know they're not implemented yet."""

    def test_benzinga_raises_on_init(self):
        with pytest.raises(NotImplementedError, match="Phase 4"):
            BenzingaNewsAdapter()

    def test_ravenpack_raises_on_init(self):
        with pytest.raises(NotImplementedError, match="Phase 4"):
            RavenpackNewsAdapter()

    def test_bloomberg_raises_on_init(self):
        with pytest.raises(NotImplementedError, match="Phase 4"):
            BloombergNewsAdapter()


# ── Aggregator: dedup + trust weighting ───────────────────────────────


class TestNormalizationHelpers:
    def test_title_normalization_lowercases_and_strips_punct(self):
        assert _normalize_title("Apple's Q4 Beat — Up 5%!") == "apple s q4 beat up 5"

    def test_title_normalization_idempotent(self):
        n = _normalize_title("Some Title!")
        assert _normalize_title(n) == n

    def test_url_fingerprint_strips_querystring(self):
        fp1 = _url_fingerprint("https://x.com/path?utm_source=a")
        fp2 = _url_fingerprint("https://x.com/path?utm_source=b&ref=c")
        assert fp1 == fp2

    def test_url_fingerprint_strips_fragment(self):
        fp1 = _url_fingerprint("https://x.com/path#section1")
        fp2 = _url_fingerprint("https://x.com/path#section2")
        assert fp1 == fp2


class TestNewsAggregatorDedup:
    def _make_static_source(self, name, articles):
        src = MagicMock(spec=["name", "fetch"])
        src.name = name
        src.fetch.return_value = articles
        return src

    def test_fan_in_concatenates_all_sources(self):
        a1 = _make_article(source="polygon", title="A")
        a2 = _make_article(source="gdelt", title="B")
        agg = NewsAggregator(sources=[
            self._make_static_source("polygon", [a1]),
            self._make_static_source("gdelt", [a2]),
        ])
        out = agg.fetch(["AAPL"])
        assert len(out) == 2
        titles = {x.canonical_title for x in out}
        assert titles == {"A", "B"}

    def test_same_url_dedups_across_sources(self):
        url = "https://reuters.com/aapl-q4-beat?utm=a"
        a1 = _make_article(source="polygon", title="Apple Q4 Beat", url=url)
        a2 = _make_article(
            source="gdelt",
            title="Apple Q4 Beat",  # same title; same URL after stripping
            url="https://reuters.com/aapl-q4-beat?utm=b",
        )
        agg = NewsAggregator(sources=[
            self._make_static_source("polygon", [a1]),
            self._make_static_source("gdelt", [a2]),
        ])
        out = agg.fetch(["AAPL"])
        assert len(out) == 1
        assert out[0].n_sources == 2
        assert {v.source for v in out[0].variants} == {"polygon", "gdelt"}

    def test_canonical_title_picks_longest(self):
        # Same normalized title, different punctuation/casing — same
        # fingerprint — picks the longest as canonical because the
        # longer form preserves more of the original publisher's
        # framing.
        url = "https://reuters.com/aapl"
        a1 = _make_article(
            source="polygon",
            title="Apple Reports Strong Q4 Beat",
            url=url,
        )
        a2 = _make_article(
            source="gdelt",
            title="Apple Reports Strong Q4 Beat!!!",
            url=url,
        )
        agg = NewsAggregator(sources=[
            self._make_static_source("polygon", [a1, a2]),
        ])
        out = agg.fetch(["AAPL"])
        assert len(out) == 1
        assert out[0].canonical_title == "Apple Reports Strong Q4 Beat!!!"

    def test_canonical_url_picks_highest_trust_source(self):
        url_polygon = "https://polygon.example/x"
        url_yahoo = "https://yahoo.example/x"
        # Different URLs but same normalized title — dedup will NOT
        # collapse them (different URL fingerprints). Use SAME url to
        # force them into one group, varying only source.
        a_polygon = _make_article(
            source="polygon", title="story", url=url_polygon
        )
        a_yahoo = _make_article(
            source="yahoo_rss", title="story", url=url_polygon,  # force same group
        )
        agg = NewsAggregator(sources=[
            self._make_static_source("polygon", [a_polygon, a_yahoo]),
        ])
        out = agg.fetch(["AAPL"])
        assert len(out) == 1
        # Polygon has higher trust weight than yahoo_rss in DEFAULT
        assert out[0].canonical_url == url_polygon
        # And the canonical_url is the polygon one despite alphabetical
        # ordering of variants

    def test_ticker_union_across_variants(self):
        url = "https://x.com/sector"
        a1 = _make_article(
            source="polygon", title="Sector roundup", url=url,
            tickers=("AAPL", "MSFT"),
        )
        a2 = _make_article(
            source="gdelt", title="Sector Roundup", url=url,
            tickers=("AAPL", "GOOGL"),
        )
        agg = NewsAggregator(sources=[
            self._make_static_source("polygon", [a1]),
            self._make_static_source("gdelt", [a2]),
        ])
        out = agg.fetch(["AAPL"])
        assert out[0].tickers == ("AAPL", "GOOGL", "MSFT")  # sorted union

    def test_one_source_raising_does_not_crash_aggregator(self):
        a_good = _make_article(source="polygon", title="ok")
        broken = MagicMock(spec=["name", "fetch"])
        broken.name = "broken_vendor"
        broken.fetch.side_effect = RuntimeError("kaboom")
        agg = NewsAggregator(sources=[
            self._make_static_source("polygon", [a_good]),
            broken,
        ])
        out = agg.fetch(["AAPL"])
        assert len(out) == 1
        assert out[0].canonical_title == "ok"

    def test_output_sorted_by_published_at_desc(self):
        old = _now() - timedelta(hours=24)
        new = _now() - timedelta(hours=2)
        a_old = _make_article(
            source="polygon", title="old", url="https://x/a", published_at=old
        )
        a_new = _make_article(
            source="polygon", title="new", url="https://x/b", published_at=new
        )
        agg = NewsAggregator(sources=[
            self._make_static_source("polygon", [a_old, a_new])
        ])
        out = agg.fetch(["AAPL"])
        assert [x.canonical_title for x in out] == ["new", "old"]

    def test_empty_fan_in_returns_empty(self):
        agg = NewsAggregator(sources=[
            self._make_static_source("polygon", []),
            self._make_static_source("gdelt", []),
        ])
        assert agg.fetch(["AAPL"]) == []


class TestNewsAggregatorTrustWeights:
    def test_default_weights_loaded(self):
        agg = NewsAggregator(sources=[])
        assert agg.trust_weight("polygon") == DEFAULT_TRUST_WEIGHTS["polygon"]
        assert agg.trust_weight("yahoo_rss") == DEFAULT_TRUST_WEIGHTS["yahoo_rss"]
        # Paid weights pinned high
        assert agg.trust_weight("bloomberg") == 1.0
        assert agg.trust_weight("ravenpack") == 1.0

    def test_custom_weights_override_defaults(self):
        agg = NewsAggregator(
            sources=[],
            trust_weights={"polygon": 0.5, "yahoo_rss": 0.95},
        )
        assert agg.trust_weight("polygon") == 0.5
        assert agg.trust_weight("yahoo_rss") == 0.95

    def test_unknown_source_defaults_to_half(self, caplog):
        agg = NewsAggregator(sources=[], trust_weights={})
        with caplog.at_level("WARNING"):
            w = agg.trust_weight("brand_new_vendor")
        assert w == 0.5
        assert any("brand_new_vendor" in r.message for r in caplog.records)


# ── Fingerprint determinism ────────────────────────────────────────────


def test_article_fingerprint_is_deterministic():
    a = _make_article(title="Stable Title", url="https://x.com/p?u=1")
    b = _make_article(title="Stable Title", url="https://x.com/p?u=2")
    assert _article_fingerprint(a) == _article_fingerprint(b)


def test_article_fingerprint_differs_on_different_titles():
    a = _make_article(title="A", url="https://x.com/p")
    b = _make_article(title="B", url="https://x.com/p")
    assert _article_fingerprint(a) != _article_fingerprint(b)
