"""Tests for the 2026-07-03 slim research-briefing email conversion
(alpha-engine-config#856 Phase 2.5 — "pull-for-state console page +
push-on-transition emails only").

Mirrors the crucible-executor EOD-email conversion (executor#276 /
dashboard#237): the full consolidated report markdown is no longer
rendered inline in the morning email. It's persisted verbatim by
``archive_writer`` via ``ArchiveManager.save_consolidated_report``
(UNTOUCHED by this conversion — still writing
``consolidated/{run_date}/morning.md``), and rendered by the dashboard's
Research Briefing Archive page. The email now carries only a compact
summary + a deep-link to that page.

Covers:
  1. The email no longer renders the full report body.
  2. The deep-link is well-formed (mirrors the eod-report `?date=`
     convention where supported; page 17 has no per-run scoping, so no
     ``?date=``/``?run=`` param is invented here — see the gap note
     above ``RESEARCH_BRIEFING_SLUG`` in graph/research_graph.py).
  3. ``save_consolidated_report`` is still called from ``archive_writer``
     with unchanged arguments (email_sender never calls it itself).
  4. No accidental behavior change to the archive-writing path.

Convention: ``monkeypatch`` fixture only — NEVER ``unittest.mock.patch``
(documented full-suite bleed via ``sys.modules`` reassignment; see
tests/test_dry_run.py / tests/test_email_redesign.py docstrings).
"""
from __future__ import annotations

import inspect
from urllib.parse import parse_qs, urlsplit

from graph.research_graph import (
    RESEARCH_BRIEFING_SLUG,
    RESEARCH_BRIEFING_TAB,
    _build_slim_briefing_email,
    _research_briefing_url,
    archive_writer,
    email_sender,
)

# A representative full consolidated report, shaped like consolidator()'s
# real output — the point is that NONE of this proprietary body text or
# section structure should leak into the slim email.
FULL_REPORT_MARKDOWN = (
    "# Daily Research Brief — 2026-07-03\n\n"
    "## a. MACRO REGIME SUMMARY\n\n"
    "Some long macro narrative that used to render fully inline in the "
    "pre-#856 email.\n\n"
    "## c. UNIVERSE RATINGS\n\n"
    "| Ticker | Status | Recommendation | Score (0-100) | Rationale |\n"
    "|--------|--------|----------------|----------------|-----------|\n"
    "| AAA | New | Buy | 88 | huge upside - proprietary rationale text |\n"
)


def _state(**overrides) -> dict:
    state = {
        "run_date": "2026-07-03",
        "market_regime": "neutral",
        "new_population": [{"ticker": "AAA"}, {"ticker": "BBB"}],
        "current_population": [{"ticker": "BBB"}],
        "exits": [{"ticker_out": "ZZZ"}],
        "consolidated_report": FULL_REPORT_MARKDOWN,
    }
    state.update(overrides)
    return state


# ── 1. No full-content rendering ────────────────────────────────────────────


class TestSlimEmailDropsFullContent:
    def test_html_body_does_not_contain_full_report_markdown(self):
        html, _ = _build_slim_briefing_email(_state())
        assert "proprietary rationale text" not in html
        assert "MACRO REGIME SUMMARY" not in html
        assert "UNIVERSE RATINGS" not in html
        # Only the small summary table remains.
        assert "<table>" in html

    def test_plain_body_does_not_contain_full_report_markdown(self):
        _, plain = _build_slim_briefing_email(_state())
        assert "proprietary rationale text" not in plain
        assert "UNIVERSE RATINGS" not in plain
        assert "MACRO REGIME SUMMARY" not in plain

    def test_summary_carries_high_signal_one_liners(self):
        html, plain = _build_slim_briefing_email(_state())
        for body in (html, plain):
            assert "2026-07-03" in body
            assert "NEUTRAL" in body
            # 2 in new_population: AAA (new, not in current_population),
            # BBB (existing); 1 exit (ZZZ).
            assert "1 new" in body
            assert "1 existing" in body
            assert "1 exited" in body

    def test_email_sender_no_longer_uses_full_markdown_formatter(self):
        src = inspect.getsource(email_sender)
        assert "format_email" not in src, (
            "email_sender must not use the full-markdown-to-HTML renderer "
            "any more (emailer.formatter.format_email is still used "
            "elsewhere, e.g. scoring/attractiveness_trajectory.py, but not "
            "for the morning briefing after config#856 Phase 2.5)."
        )

    def test_email_sender_does_not_call_save_consolidated_report_itself(self):
        # Persistence is archive_writer's job, unconditionally, before
        # email_sender ever runs. email_sender must not duplicate it.
        src = inspect.getsource(email_sender)
        # The docstring mentions the persistence chokepoint by name for
        # context; what must not appear is an actual CALL to it.
        assert ".save_consolidated_report(" not in src


# ── 2. Deep-link shape ───────────────────────────────────────────────────────


class TestDeepLink:
    def test_link_targets_the_documented_host_tab(self):
        url = _research_briefing_url()
        parsed = urlsplit(url)
        assert parsed.scheme == "https"
        assert parsed.path.strip("/") == RESEARCH_BRIEFING_SLUG
        qs = parse_qs(parsed.query)
        assert qs.get("tab") == [RESEARCH_BRIEFING_TAB]

    def test_base_url_override_respected(self):
        url = _research_briefing_url("https://console.example.com/")
        parsed = urlsplit(url)
        assert parsed.netloc == "console.example.com"
        assert parsed.path.strip("/") == RESEARCH_BRIEFING_SLUG

    def test_no_per_run_date_or_run_query_param(self):
        # Deliberate: unlike .../eod-report?date=YYYY-MM-DD, page 17 has no
        # per-run selection support (see the gap note in
        # graph/research_graph.py above RESEARCH_BRIEFING_SLUG) — it always
        # shows "latest inline + prior weeks click-to-expand". This link
        # must not invent a ?date=/?run= param the page would silently
        # ignore.
        url = _research_briefing_url()
        qs = parse_qs(urlsplit(url).query)
        assert "date" not in qs
        assert "run" not in qs

    def test_deep_link_included_in_both_email_bodies(self):
        html, plain = _build_slim_briefing_email(_state())
        url = _research_briefing_url()
        assert url in html
        assert url in plain


# ── 3 & 4. save_consolidated_report untouched / archive path unaffected ────


class TestArchiveWritingPathUnaffected:
    def test_archive_writer_still_calls_save_consolidated_report_unchanged(self):
        # Pin the exact call shape archive_writer uses to persist the full
        # report — this conversion must not touch it. Regression coverage
        # complementing tests/test_archive.py::
        # TestConsolidatedReportPersistence (2026-05-20 silent-drop fix).
        src = inspect.getsource(archive_writer)
        assert "am.save_consolidated_report(run_date, consolidated)" in src

    def test_save_consolidated_report_receives_full_report_not_slim_summary(self):
        # Guard against a future refactor accidentally wiring the SLIM
        # summary into the archive write path instead of the full report —
        # the console archive page must keep rendering the complete brief.
        # archive_writer is a large function with many side effects; a full
        # integration run is out of scope here (see test_archive.py /
        # test_archive_writer_signals_contract.py for that coverage). This
        # test only pins that email_sender's slim summary and
        # archive_writer's persisted artifact are DIFFERENT strings drawn
        # from the same `consolidated_report` state field, i.e. slimming
        # the email must not slim the archive.
        state = _state()
        html, plain = _build_slim_briefing_email(state)
        assert state["consolidated_report"] not in html
        assert state["consolidated_report"] not in plain
        # The full report is exactly what save_consolidated_report expects
        # to persist (archive_writer passes `consolidated` = the raw
        # state field, verbatim).
        assert state["consolidated_report"] == FULL_REPORT_MARKDOWN


# ── End-to-end email_sender wiring ──────────────────────────────────────────


class TestEmailSenderWiring:
    def test_sends_slim_body_via_emailer_sender_send_email(self, monkeypatch):
        sent = {}

        def _fake_send_email(*, subject, html_body, plain_body, recipients, sender):
            sent["subject"] = subject
            sent["html_body"] = html_body
            sent["plain_body"] = plain_body
            sent["recipients"] = recipients
            sent["sender"] = sender

        import config
        import emailer.sender as sender_mod

        monkeypatch.setattr(sender_mod, "send_email", _fake_send_email)
        monkeypatch.setattr(config, "EMAIL_RECIPIENTS", ["ops@example.com"])
        monkeypatch.setattr(config, "EMAIL_SENDER", "bot@example.com")

        result = email_sender(_state())

        assert result == {"email_sent": True}
        assert sent["recipients"] == ["ops@example.com"]
        assert sent["sender"] == "bot@example.com"
        assert "2026-07-03" in sent["subject"]
        assert "proprietary rationale text" not in sent["html_body"]
        assert "proprietary rationale text" not in sent["plain_body"]
        assert _research_briefing_url() in sent["html_body"]

    def test_no_consolidated_report_skips_send(self, monkeypatch):
        called = []
        import emailer.sender as sender_mod
        monkeypatch.setattr(
            sender_mod, "send_email", lambda **kw: called.append(kw)
        )
        result = email_sender(_state(consolidated_report=""))
        assert result == {"email_sent": False}
        assert called == []

    def test_send_failure_is_caught_and_reported(self, monkeypatch):
        import config
        import emailer.sender as sender_mod

        def _boom(**kw):
            raise RuntimeError("smtp down")

        monkeypatch.setattr(sender_mod, "send_email", _boom)
        monkeypatch.setattr(config, "EMAIL_RECIPIENTS", ["ops@example.com"])
        monkeypatch.setattr(config, "EMAIL_SENDER", "bot@example.com")

        result = email_sender(_state())
        assert result == {"email_sent": False}
