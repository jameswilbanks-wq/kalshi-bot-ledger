#!/usr/bin/env python3
"""
Standalone unit test for the ESPN sports-odds fallback added 2026-09-16.

Follows the same pattern as the project's existing test_same_event_guard.py
and test_sports_throttle.py: no network, no real credentials -- imports the
real module with requests.get replaced by a fake, and asserts on behavior.

Covers:
  1. Off-hour gating -- _fetch_espn_scoreboard makes ZERO network calls
     outside SPORTS_CHECK_HOURS_UTC, same as the primary Odds API path.
  2. Happy path -- sports_fair_value() falls back to ESPN and returns the
     correct no-vig probability when the Odds API has no key configured
     (the exact failure mode that triggered building this).
  3. +1 day date slack -- a game ESPN files under game_date+1 is still
     found.
  4. Failure caching -- a scoreboard request that raises is only ever
     attempted ONCE per (sport, date) per run, not once per market.
  5. Unmapped sport -- a sport_key with no ESPN_SPORT_PATHS entry never
     triggers a network call.

Run: python3 test_espn_sports_fallback.py
"""
import datetime as dt
import os
import sys
import unittest
from unittest import mock

# Required at import time (kalshi_edge_bot.py reads this eagerly) -- fake
# value only, this test never makes a real Kalshi call.
os.environ.setdefault("KALSHI_API_KEY_ID", "test-key-id")
os.environ["ODDS_API_KEY"] = ""  # force the primary path to return None with zero network calls

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalshi_edge_bot as bot  # noqa: E402


def _reset_caches():
    bot._espn_scoreboard_cache.clear()
    bot._espn_scoreboard_failed_this_tick.clear()
    bot._espn_odds_cache.clear()
    bot._espn_odds_failed_this_tick.clear()
    bot._odds_api_cache.clear()
    bot._odds_api_failed_this_tick.clear()


def _frozen_utcnow(hour):
    class _FakeDatetime(dt.datetime):
        @classmethod
        def utcnow(cls):
            return dt.datetime(2026, 9, 20, hour, 0, 0)
    return _FakeDatetime


SCOREBOARD_CHI_HOME = {
    "events": [{
        "id": "999001",
        "competitions": [{
            "competitors": [
                {"homeAway": "home", "team": {"displayName": "Chicago Bears"}},
                {"homeAway": "away", "team": {"displayName": "Minnesota Vikings"}},
            ]
        }],
    }]
}

SUMMARY_WITH_ODDS = {
    "pickcenter": [{
        "provider": {"name": "DraftKings"},
        "homeTeamOdds": {"moneyLine": -150},
        "awayTeamOdds": {"moneyLine": 130},
    }]
}

EMPTY_SCOREBOARD = {"events": []}


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class TestEspnFallback(unittest.TestCase):
    def setUp(self):
        _reset_caches()

    # 1. Off-hour gating -------------------------------------------------
    def test_off_hour_makes_zero_calls(self):
        with mock.patch.object(bot.dt, "datetime", _frozen_utcnow(1)):  # 01:00 UTC -- not in SPORTS_CHECK_HOURS_UTC
            with mock.patch.object(bot.requests, "get") as mock_get:
                result = bot._fetch_espn_scoreboard("americanfootball_nfl", "20260920")
        self.assertIsNone(result)
        mock_get.assert_not_called()

    # 2. Happy path via the real public entry point ----------------------
    def test_sports_fair_value_falls_back_to_espn(self):
        market = {
            "ticker": "KXNFLGAME-26SEP20MINCHI-CHI",
            "event_ticker": "KXNFLGAME-26SEP20MINCHI",
        }

        calls = []

        def fake_get(url, params=None, timeout=10):
            calls.append((url, dict(params or {})))
            if url.endswith("/scoreboard"):
                return FakeResponse(SCOREBOARD_CHI_HOME)
            if url.endswith("/summary"):
                return FakeResponse(SUMMARY_WITH_ODDS)
            return FakeResponse({}, status=500)

        with mock.patch.object(bot.dt, "datetime", _frozen_utcnow(12)):  # in SPORTS_CHECK_HOURS_UTC
            with mock.patch.object(bot.requests, "get", side_effect=fake_get):
                prob = bot.sports_fair_value(market)

        # Must match on the FIRST candidate date and stop -- no wasted
        # second scoreboard call once offset=0 already found the game.
        scoreboard_calls = [c for c in calls if c[0].endswith("/scoreboard")]
        self.assertEqual(len(scoreboard_calls), 1)
        self.assertEqual(scoreboard_calls[0][1].get("dates"), "20260920")
        summary_calls = [c for c in calls if c[0].endswith("/summary")]
        self.assertEqual(len(summary_calls), 1)
        self.assertEqual(summary_calls[0][1].get("event"), "999001")

        # implied(home, -150) = 150/250 = 0.60; implied(away, +130) = 100/230 ≈ 0.43478
        # no-vig home = 0.60 / (0.60 + 0.43478) ≈ 0.5798
        self.assertIsNotNone(prob)
        self.assertAlmostEqual(prob, 0.5798, places=3)

    # 3. +1 day slack ------------------------------------------------------
    def test_plus_one_day_slack(self):
        calls = []

        def fake_get(url, params=None, timeout=10):
            calls.append((url, dict(params or {})))
            if url.endswith("/scoreboard"):
                if params.get("dates") == "20260920":
                    return FakeResponse(EMPTY_SCOREBOARD)  # nothing on the exact date
                if params.get("dates") == "20260921":
                    return FakeResponse(SCOREBOARD_CHI_HOME)  # ESPN filed it a day later (UTC)
                return FakeResponse({}, status=500)
            if url.endswith("/summary"):
                return FakeResponse(SUMMARY_WITH_ODDS)
            return FakeResponse({}, status=500)

        with mock.patch.object(bot.dt, "datetime", _frozen_utcnow(12)):
            with mock.patch.object(bot.requests, "get", side_effect=fake_get):
                prob = bot.fetch_espn_nfl_fair_prob(
                    "americanfootball_nfl", dt.date(2026, 9, 20), "MIN", "CHI", yes_is_home=True
                )
        self.assertIsNotNone(prob)
        self.assertAlmostEqual(prob, 0.5798, places=3)
        scoreboard_dates = [c[1].get("dates") for c in calls if c[0].endswith("/scoreboard")]
        self.assertEqual(scoreboard_dates, ["20260920", "20260921"], "must try offset 0 before offset 1")

    # 4. Failure caching -----------------------------------------------
    def test_scoreboard_failure_cached_per_tick(self):
        call_count = {"n": 0}

        def fake_get(url, params=None, timeout=10):
            call_count["n"] += 1
            raise Exception("simulated network failure")

        with mock.patch.object(bot.dt, "datetime", _frozen_utcnow(12)):
            with mock.patch.object(bot.requests, "get", side_effect=fake_get):
                r1 = bot._fetch_espn_scoreboard("americanfootball_nfl", "20260920")
                r2 = bot._fetch_espn_scoreboard("americanfootball_nfl", "20260920")  # same key -- should NOT re-hit
        self.assertIsNone(r1)
        self.assertIsNone(r2)
        self.assertEqual(call_count["n"], 1, "a failed scoreboard call must only be attempted once per tick")

    # 5. Unmapped sport never calls out -----------------------------------
    def test_unmapped_sport_makes_zero_calls(self):
        with mock.patch.object(bot.dt, "datetime", _frozen_utcnow(12)):
            with mock.patch.object(bot.requests, "get") as mock_get:
                result = bot._fetch_espn_scoreboard("basketball_nba", "20260920")
        self.assertIsNone(result)
        mock_get.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
