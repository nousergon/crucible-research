"""Tests for shared JSON extraction utilities."""

from agents.json_utils import extract_json_array, extract_json_object


class TestExtractJsonObject:
    def test_extracts_from_mixed_text(self):
        text = 'Here is my analysis:\n{"score": 75, "rating": "BUY"}\nThat is all.'
        result = extract_json_object(text)
        assert result == {"score": 75, "rating": "BUY"}

    def test_extracts_with_hint_key(self):
        text = '{"other": 1} some text {"market_regime": "bull", "vix": 18.5}'
        result = extract_json_object(text, hint_key='"market_regime"')
        assert result["market_regime"] == "bull"

    def test_handles_nested_objects(self):
        text = '{"outer": {"inner": {"deep": true}}, "value": 42}'
        result = extract_json_object(text)
        assert result["outer"]["inner"]["deep"] is True
        assert result["value"] == 42

    def test_handles_escaped_quotes(self):
        text = '{"text": "He said \\"hello\\"", "score": 5}'
        result = extract_json_object(text)
        assert result["score"] == 5

    def test_returns_none_on_no_json(self):
        result = extract_json_object("No JSON here at all.")
        assert result is None

    def test_returns_none_on_malformed_json(self):
        result = extract_json_object('{"unclosed": "brace')
        assert result is None


class TestExtractJsonArray:
    def test_extracts_array(self):
        text = 'Results: [{"ticker": "AAPL"}, {"ticker": "MSFT"}]'
        result = extract_json_array(text)
        assert len(result) == 2
        assert result[0]["ticker"] == "AAPL"

    def test_fallback_to_individual_objects(self):
        text = 'Pick 1: {"ticker": "AAPL", "score": 80}\nPick 2: {"ticker": "MSFT", "score": 75}'
        result = extract_json_array(text)
        assert len(result) == 2

    def test_returns_none_on_no_json(self):
        result = extract_json_array("No arrays here.")
        assert result is None

    def test_fallback_preserves_nested_objects(self):
        # Regression: old regex r'\{[^{}]+\}' matched only innermost objects,
        # dropping the outer pick with its catalyst sub-object. Simulate a
        # malformed array (trailing commas) that trips the array parser.
        text = (
            'Picks:\n'
            '{"ticker": "AAPL", "catalyst": {"reason": "strong"}},\n'
            '{"ticker": "MSFT", "catalyst": {"reason": "stable"}},\n'
        )
        result = extract_json_array(text)
        assert result is not None
        assert {o["ticker"] for o in result if "ticker" in o} == {"AAPL", "MSFT"}

    def test_require_key_filters_inner_objects(self):
        # When we require a "ticker" key, the nested catalyst sub-object is
        # dropped, leaving only the outer picks. This is what the sector-team
        # analysts use.
        text = (
            '{"ticker": "AAPL", "catalyst": {"reason": "strong"}},\n'
            '{"ticker": "MSFT", "catalyst": {"reason": "stable"}},\n'
        )
        result = extract_json_array(text, require_key="ticker")
        assert result is not None
        assert len(result) == 2
        assert all("ticker" in o for o in result)
