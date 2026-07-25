"""
Tests for trading day detection and handler gate logic.
Validates scheduler logic without triggering real AWS or LLM calls.
"""

import datetime
import importlib
from unittest.mock import patch

# 'lambda' is a reserved keyword; use importlib to load lambda.handler
_lambda_handler = importlib.import_module("lambda.handler")


class TestTradingDayDetection:
    def test_weekday_is_trading_day(self):
        is_trading_day = _lambda_handler.is_trading_day
        # 2026-03-04 is a Wednesday
        d = datetime.date(2026, 3, 4)
        assert is_trading_day(d) is True

    def test_weekend_is_not_trading_day(self):
        is_trading_day = _lambda_handler.is_trading_day
        # 2026-03-07 is a Saturday
        d = datetime.date(2026, 3, 7)
        assert is_trading_day(d) is False

    def test_christmas_is_not_trading_day(self):
        is_trading_day = _lambda_handler.is_trading_day
        # 2025-12-25 is Christmas (NYSE closed)
        d = datetime.date(2025, 12, 25)
        assert is_trading_day(d) is False

    def test_mlk_day_is_not_trading_day(self):
        is_trading_day = _lambda_handler.is_trading_day
        # 2026-01-19 is MLK Day
        d = datetime.date(2026, 1, 19)
        assert is_trading_day(d) is False


class TestHandlerHolidaySkip:
    @patch("lambda.handler._is_scheduled_run_time", return_value=True)
    @patch("lambda.handler.is_trading_day", return_value=False)
    def test_skips_on_holiday(self, mock_trading_day, mock_time):
        handler = _lambda_handler.handler
        result = handler({}, {})
        assert result["status"] == "SKIPPED"
        assert result["reason"] == "market_holiday"
