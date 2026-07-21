"""Tests for material trigger detection — determines when held stock theses need updating."""

from agents.sector_teams.material_triggers import check_material_triggers


class TestNewsTrigger:
    def test_news_spike_triggers(self):
        triggers = check_material_triggers(
            ticker="AAPL",
            news_data={"articles": [{"headline": f"h{i}"} for i in range(5)], "article_count": 5},
            price_data=None,
            analyst_data=None,
            insider_data=None,
            prior_thesis={"score": 70},
            sector_regime_changed=False,
            run_date="2026-03-26",
        )
        assert triggers is not None
        assert any("news" in str(t).lower() for t in triggers)

    def test_no_news_no_trigger(self):
        triggers = check_material_triggers(
            ticker="AAPL",
            news_data={"articles": [], "article_count": 0},
            price_data=None,
            analyst_data=None,
            insider_data=None,
            prior_thesis={"score": 70},
            sector_regime_changed=False,
            run_date="2026-03-26",
        )
        assert not triggers


class TestSectorRegimeTrigger:
    def test_regime_change_triggers(self):
        triggers = check_material_triggers(
            ticker="AAPL",
            news_data=None,
            price_data=None,
            analyst_data=None,
            insider_data=None,
            prior_thesis={"score": 70},
            sector_regime_changed=True,
            run_date="2026-03-26",
        )
        assert triggers is not None
        assert any("regime" in str(t).lower() for t in triggers)


class TestNoTrigger:
    def test_all_none_no_trigger(self):
        triggers = check_material_triggers(
            ticker="AAPL",
            news_data=None,
            price_data=None,
            analyst_data=None,
            insider_data=None,
            prior_thesis={"score": 70},
            sector_regime_changed=False,
            run_date="2026-03-26",
        )
        assert not triggers

    def test_no_prior_thesis_no_trigger(self):
        triggers = check_material_triggers(
            ticker="AAPL",
            news_data=None,
            price_data=None,
            analyst_data=None,
            insider_data=None,
            prior_thesis=None,
            sector_regime_changed=False,
            run_date="2026-03-26",
        )
        assert not triggers
