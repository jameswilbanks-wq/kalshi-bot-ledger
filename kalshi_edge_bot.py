#!/usr/bin/env python3
"""
kalshi_edge_bot.py

A small, honest version of the "Grok Bot" strategy shown in the video, built
against Kalshi's real API. One run of main() = one "tick" of the loop:

    scan markets -> estimate fair value -> flag mispricing > threshold
    -> size with Kelly criterion (capped) -> place order -> log it

Run it by hand, on a cron, or deploy it as a scheduled job (e.g. a Render
Cron Job) that fires every 10 minutes -- that's what makes it "autonomous."

BEFORE YOU RUN THIS FOR REAL:
  1. Set KALSHI_ENV=demo and confirm everything works against Kalshi's demo
     (paper) environment for at least several days.
  2. Read the fair-value section below. Only the weather model is a real,
     data-backed edge. Every other category returns "no opinion" (skip) by
     design -- do NOT invent a fake sentiment score just to have more trades.
  3. Only flip KALSHI_ENV=prod and LIVE_TRADING=true once you've watched the
     paper results and are comfortable with them.

Dependencies:
    pip install requests cryptography
"""

import os
import csv
import time
import base64
import logging
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

load_dotenv()  # reads the .env file in this folder and loads it into os.environ

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("edge_bot")

# --------------------------------------------------------------------------
# CONFIG -- everything here is read from environment variables so no
# secrets ever live in this file. Set these in a .env file locally, or as
# Render environment variables when you deploy the cron job.
# --------------------------------------------------------------------------

KALSHI_ENV = os.environ.get("KALSHI_ENV", "demo")            # "demo" or "prod"
KALSHI_API_KEY_ID = os.environ["KALSHI_API_KEY_ID"]
# Local runs: set KALSHI_PRIVATE_KEY_PATH to a file. Cloud runs: set
# KALSHI_PRIVATE_KEY_PEM to the key's actual contents instead (see KalshiClient).
KALSHI_PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")

LIVE_TRADING = os.environ.get("LIVE_TRADING", "false").lower() == "true"

EDGE_THRESHOLD = float(os.environ.get("EDGE_THRESHOLD", "0.08"))      # 8%, per the video
MAX_KELLY_FRACTION = float(os.environ.get("MAX_KELLY_FRACTION", "0.06"))  # 6% of bankroll cap
KELLY_MULTIPLIER = float(os.environ.get("KELLY_MULTIPLIER", "0.5"))   # "half Kelly" is safer than full Kelly
MAX_DAILY_LOSS_PCT = float(os.environ.get("MAX_DAILY_LOSS_PCT", "0.15"))  # kill switch: stop for the day at -15%
MIN_CONTRACT_VOLUME = int(os.environ.get("MIN_CONTRACT_VOLUME", "50"))    # skip illiquid/thin markets
LEDGER_PATH = Path(os.environ.get("LEDGER_PATH", "ledger.csv"))

# Confirmed live on 2026-09-09: Kalshi's demo/sandbox markets carry
# volume_fp=0.00 across the board -- there are no real counterparties
# trading against a paper account, so "volume" there isn't a liquidity
# signal the way it is in prod, it's just always ~zero. Enforcing the same
# floor in demo silently vetoes almost every market for a reason that has
# nothing to do with the strategy (64/66 markets failed here in one
# observed tick). Only enforce the floor where it's protecting against a
# real, costly risk: trusting a price nobody has actually traded at.
EFFECTIVE_MIN_VOLUME = MIN_CONTRACT_VOLUME if KALSHI_ENV == "prod" else 0

# Optional: publish the ledger to a small PUBLIC GitHub repo after every
# tick, so a dashboard hosted anywhere can read it with no auth. This repo
# should contain ONLY this JSON file -- never your strategy code or keys.
# If these three aren't all set, the bot just skips this step silently and
# behaves exactly as before (local CSV logging only).
GITHUB_LEDGER_TOKEN = os.environ.get("GITHUB_LEDGER_TOKEN", "")
GITHUB_LEDGER_REPO = os.environ.get("GITHUB_LEDGER_REPO", "")   # e.g. "jameswilbanks-wq/kalshi-bot-ledger"
GITHUB_LEDGER_PATH = os.environ.get("GITHUB_LEDGER_PATH", "ledger.json")
MAX_LEDGER_DECISIONS = 500    # keep the public file small; older history stays in the private CSV
MAX_LEDGER_BANKROLL_POINTS = 2000
MAX_PROD_OBSERVATIONS = 2000  # capped history of real-market edge sightings, for later calibration

# Optional: financials/economics model (Alpha Vantage) and sports model
# (The Odds API). Both are None-by-default -- if a key isn't set, that
# category's fair_value function returns None for every market, same as
# any other model we don't have real data for. No fake signal either way.
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# Alpha Vantage's free tier is 25 requests PER DAY, not per minute. This
# bot ticks every 10 minutes (144x/day) -- calling Alpha Vantage on every
# tick would blow the daily quota before lunch. So financial quotes are
# only actually fetched during a few specific UTC hours (roughly US
# market open/midday/close), and economic data (which only changes on
# scheduled monthly release dates anyway) only once a day. Outside those
# windows, the functions below return None and that tick just skips the
# category -- exactly like "no model for this market," not an error.
# This means the financial fair-value number can be a few hours stale at
# times. That's a real tradeoff of the free tier, not a bug -- if this
# model earns its keep, a paid tier removes the constraint.
FINANCIAL_CHECK_HOURS_UTC = {14, 17, 20}   # ~9:30am/12:30pm/3:30pm ET, roughly open/mid/close
ECONOMIC_CHECK_HOUR_UTC = 6                # once a day is plenty; these change monthly at most

# NOTE: Kalshi has changed its base API host more than once. Verify the
# current value at https://docs.kalshi.com before relying on this.
BASE_URLS = {
    "demo": "https://external-api.demo.kalshi.co/trade-api/v2",
    "prod": "https://external-api.kalshi.com/trade-api/v2",
}
BASE_URL = BASE_URLS[KALSHI_ENV]

# Kalshi's PRODUCTION market-data endpoints (GetMarkets, orderbook, etc.)
# are public and require no API key or funded account -- confirmed via
# Kalshi's own docs (docs.kalshi.com/getting_started/quick_start_market_data).
# This is used ONLY to observe real prices/volume for calibration, always
# against real prod data regardless of KALSHI_ENV, and NEVER to place an
# order -- see observe_prod_edges() below.
PROD_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"


# --------------------------------------------------------------------------
# KALSHI CLIENT -- RSA-PSS signed requests, per Kalshi's auth spec.
# --------------------------------------------------------------------------

class KalshiClient:
    def __init__(self, key_id: str, private_key_path: str, base_url: str):
        self.key_id = key_id
        self.base_url = base_url
        # Three ways the private key might be provided, checked in order:
        #  1. KALSHI_PRIVATE_KEY_PEM env var -- the raw key text directly.
        #  2. Render's "Secret Files" feature -- appears on disk at
        #     /etc/secrets/<filename>, which is actually Render's own
        #     recommended way to store something like a private key.
        #  3. A plain local file path (KALSHI_PRIVATE_KEY_PATH) -- what
        #     local/Windows runs use.
        pem_from_env = os.environ.get("KALSHI_PRIVATE_KEY_PEM")
        secret_file_path = Path("/etc/secrets/KALSHI_PRIVATE_KEY_PEM")
        if pem_from_env:
            key_bytes = pem_from_env.encode("utf-8")
        elif secret_file_path.exists():
            key_bytes = secret_file_path.read_bytes()
        else:
            with open(private_key_path, "rb") as f:
                key_bytes = f.read()
        self.private_key = serialization.load_pem_private_key(key_bytes, password=None)

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        message = f"{timestamp_ms}{method}{path}".encode("utf-8")
        signature = self.private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _headers(self, method: str, path_for_signature: str) -> dict:
        timestamp_ms = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": self._sign(timestamp_ms, method, path_for_signature),
            "Content-Type": "application/json",
        }

    def get(self, path: str, params: Optional[dict] = None) -> dict:
        # Kalshi signs the path WITHOUT the query string.
        resp = requests.get(
            self.base_url + path, params=params, headers=self._headers("GET", "/trade-api/v2" + path)
        )
        resp.raise_for_status()
        return resp.json()

    def post(self, path: str, body: dict) -> dict:
        resp = requests.post(
            self.base_url + path, json=body, headers=self._headers("POST", "/trade-api/v2" + path)
        )
        resp.raise_for_status()
        return resp.json()

    def get_balance_cents(self) -> int:
        return self.get("/portfolio/balance")["balance"]

    def get_open_markets(self, limit: int = 200) -> list:
        markets, cursor = [], None
        while len(markets) < limit:
            params = {"status": "open", "limit": min(200, limit - len(markets))}
            if cursor:
                params["cursor"] = cursor
            page = self.get("/markets", params=params)
            markets.extend(page.get("markets", []))
            cursor = page.get("cursor")
            if not cursor:
                break
        return markets

    def get_markets_by_series(self, series_ticker: str) -> list:
        """Ask Kalshi directly for one series' open markets, instead of
        hoping it shows up in an arbitrary top-N scan. This is the approach
        Kalshi's own docs recommend -- much more reliable than guessing
        whether a category appears in a random sample."""
        markets, cursor = [], None
        while True:
            params = {"series_ticker": series_ticker, "status": "open", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            page = self.get("/markets", params=params)
            markets.extend(page.get("markets", []))
            cursor = page.get("cursor")
            if not cursor:
                break
        return markets

    def get_known_weather_markets(self) -> list:
        """Query every series we have a model for (see NWS_GRIDPOINTS)
        directly by name, rather than scanning everything and filtering."""
        return self.get_known_markets(NWS_GRIDPOINTS)

    def get_known_markets(self, series_registry: dict) -> list:
        """Generalized version of get_known_weather_markets -- query every
        series ticker in the given registry (keys are Kalshi series
        tickers; values are whatever that model needs, e.g. coordinates,
        a proxy symbol, a sport key -- this method doesn't care). A wrong
        or retired ticker just returns zero markets for that key and logs
        a warning; it never crashes the tick."""
        markets = []
        for series_ticker in series_registry:
            try:
                markets.extend(self.get_markets_by_series(series_ticker))
            except requests.exceptions.HTTPError as e:
                log.warning(f"series {series_ticker} lookup failed (may not exist): {e}")
        return markets

    def place_order(self, ticker: str, side: str, count: int, price_cents: int, action: str = "buy") -> dict:
        body = {
            "ticker": ticker,
            "client_order_id": f"edgebot-{ticker}-{int(time.time())}",
            "side": side,          # "yes" or "no"
            "action": action,      # "buy" or "sell"
            "count": count,
            "type": "limit",
            "yes_price" if side == "yes" else "no_price": price_cents,
        }
        return self.post("/portfolio/orders", body)


# --------------------------------------------------------------------------
# FAIR VALUE MODELS
#
# This is the part of the viral video that's mostly theater ("it reads
# every post on X" -- no fabricated sentiment score here). Every model
# below only fires where there's an *actual* independent data source to
# compare Kalshi's price against; everything else returns None ("no
# opinion") and that market is skipped, not guessed at. Four categories,
# roughly strongest-to-weakest evidence:
#
#   - Weather: real NWS forecast vs. Kalshi's price. Well-known, legitimate
#     edge -- market-implied probabilities on weather contracts don't
#     always track the latest official forecast.
#
#   - Financials: a live ETF-proxy quote (Alpha Vantage) plus an assumed
#     volatility, same math as option pricing. Real, if simplified.
#
#   - Sports: no-vig consensus probability from real sportsbook moneylines
#     (The Odds API) vs. Kalshi's price -- the classic "compare a soft
#     line to sharp ones" edge professional bettors use.
#
#   - Economics: CPI trend extrapolation vs. Kalshi's threshold. Weakest of
#     the four -- Alpha Vantage gives past releases, not analyst consensus
#     for the next one, so this is closer to an informed guess than a
#     real forecast. Treat any edge here with extra skepticism.
#
#   - Anything else: returns None. Wire in your own model here before
#     trusting a new category (e.g. an xAI Grok "live search" call, or
#     your own view on an event) -- do not fabricate a signal to fill a
#     category out.
# --------------------------------------------------------------------------

# Kalshi city-temperature market series -> NWS gridpoint forecast endpoint.
# KXHIGHNY = Kalshi's own docs example (NYC). KXTEMPMIAH = confirmed against
# a live scan on 2026-09-07 (Miami). Kalshi does NOT use one consistent
# naming pattern across cities (their own glossary warns against assuming
# one) -- get_known_weather_markets() queries each of these directly, so a
# wrong/retired ticker here just returns zero markets, it won't crash.
# Kalshi city-temperature market series -> approximate city lat/lon. We let
# NWS resolve the actual gridpoint from these coordinates (via the /points
# endpoint) instead of hardcoding gridpoint office/x/y ourselves -- that's
# what caused the NYC lookup to fail before (a guessed, invalid gridpoint).
# Kalshi does NOT use one consistent ticker naming pattern across cities
# (their own glossary warns against assuming one) -- get_known_weather_markets()
# queries each of these directly, so a wrong/retired ticker just returns zero
# markets, it won't crash.
CITY_COORDS = {
    "KXHIGHNY": (40.7829, -73.9654),    # NYC (Central Park)
    "KXTEMPMIAH": (25.7617, -80.1918),  # Miami
    "KXTEMPCHIH": (41.8781, -87.6298),  # Chicago
    "KXHIGHCHI": (41.8781, -87.6298),   # Chicago (alt ticker pattern)
    "KXTEMPAUSH": (30.2672, -97.7431),  # Austin
    "KXHIGHAUS": (30.2672, -97.7431),   # Austin (alt ticker pattern)
}
NWS_GRIDPOINTS = CITY_COORDS  # kept for backwards-compat with earlier code/messages

# Kalshi's daily "will <index> close above/below X today" series -> a
# live ETF proxy Alpha Vantage's free GLOBAL_QUOTE endpoint can actually
# quote (Alpha Vantage has no free raw-index feed). KXNASDAQDUD confirmed
# via a live Kalshi search on 2026-09-10. Kalshi resolves against the
# ACTUAL index (per its docs, via a source like Google Finance), so QQQ's
# small tracking difference from raw NASDAQ-100 is a real, if minor,
# source of model error -- not eliminated, just disclosed.
FINANCIAL_SERIES = {
    "KXNASDAQDUD": "QQQ",   # Nasdaq-100 daily up/down
}

# Kalshi's CPI print-threshold series -> the Alpha Vantage economic
# function used to build a trend estimate. Both series confirmed via a
# live Kalshi search on 2026-09-10 (KXCPI = month-over-month, KXCPIYOY =
# year-over-year). IMPORTANT CAVEAT, unlike weather: Alpha Vantage gives
# actual PAST releases, not analyst consensus for the upcoming one, so
# economic_fair_value() below is a trend extrapolation, not "compare to
# an expert forecaster." Treat it as meaningfully weaker evidence than
# weather or financials -- it's closer to an informed guess than either.
ECONOMIC_SERIES = {
    "KXCPI": "CPI",
    "KXCPIYOY": "CPI",   # same underlying series; the model reads YoY either way
}

# Kalshi's NFL game-winner series -> The Odds API's sport key. Ticker and
# event-ticker date+team-code format confirmed via a live Kalshi search
# on 2026-09-10 (e.g. KXNFLGAME-26SEP10SFLAR = SF at LA Rams, Sep 10).
SPORTS_SERIES = {
    "KXNFLGAME": "americanfootball_nfl",
}

# Kalshi's NFL event tickers encode teams as 2-3 letter codes concatenated
# (away+home), e.g. "...-SFLAR" = SF at LAR. The Odds API reports full
# team names ("San Francisco 49ers"). This maps Odds-API full names to
# the exact codes Kalshi uses, so the two can be matched by date+teams.
# Confirmed against real Kalshi event tickers seen in search results;
# not yet confirmed against every possible team pairing -- a mismatch
# here just means that game is skipped (no crash), same failure mode as
# every other "we don't recognize this one" case in this bot.
NFL_TEAM_CODES = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

_forecast_cache: dict = {}  # (lat, lon) -> list of NWS forecast periods, per run


def _get_forecast_periods(lat: float, lon: float) -> Optional[list]:
    key = (lat, lon)
    if key in _forecast_cache:
        return _forecast_cache[key]
    headers = {"User-Agent": "edge-bot (personal project, contact: n/a)"}
    try:
        point = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=headers, timeout=10).json()
        forecast_url = point["properties"]["forecast"]
        forecast = requests.get(forecast_url, headers=headers, timeout=10).json()
        periods = forecast["properties"]["periods"]
    except Exception as e:
        log.warning(f"NWS lookup failed for ({lat},{lon}): {e}")
        return None
    _forecast_cache[key] = periods
    return periods


def _parse_target_datetime(event_ticker: str) -> tuple:
    """Kalshi event tickers end in a date code like '26SEP08' (a whole-day
    market -- daily max temperature) or '26SEP0716' (date + settlement
    HOUR -- temperature AT that specific hour, a very different question).
    Returns (date, hour_or_None)."""
    import re
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})(\d{2})?$", event_ticker)
    if not m:
        return None, None
    yy, mon_abbr, dd, hh = m.groups()
    try:
        month = dt.datetime.strptime(mon_abbr, "%b").month
        target_date = dt.date(2000 + int(yy), month, int(dd))
    except ValueError:
        return None, None
    return target_date, (int(hh) if hh is not None else None)


def _get_hourly_forecast_periods(lat: float, lon: float) -> Optional[list]:
    key = (lat, lon, "hourly")
    if key in _forecast_cache:
        return _forecast_cache[key]
    headers = {"User-Agent": "edge-bot (personal project, contact: n/a)"}
    try:
        point = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=headers, timeout=10).json()
        forecast_url = point["properties"]["forecastHourly"]
        forecast = requests.get(forecast_url, headers=headers, timeout=10).json()
        periods = forecast["properties"]["periods"]
    except Exception as e:
        log.warning(f"NWS hourly lookup failed for ({lat},{lon}): {e}")
        return None
    _forecast_cache[key] = periods
    return periods


def weather_fair_value(market: dict) -> Optional[float]:
    """Return an estimated P(YES) for a Kalshi temperature-threshold market,
    using the NWS forecast for the SPECIFIC date (and, for "at a given hour"
    markets, the specific hour) the market resolves on."""
    event_ticker = market.get("event_ticker", "")
    coords = next((c for prefix, c in CITY_COORDS.items() if event_ticker.startswith(prefix)), None)
    if not coords:
        return None

    target_date, target_hour = _parse_target_datetime(event_ticker)
    if target_date is None:
        return None

    if target_date <= dt.date.today():
        # Same reasoning as the same-day "at a specific hour" markets above:
        # if the target day has already started, today's actual conditions
        # are partially observed and the market almost certainly reflects
        # that, while this bot only has a forecast made earlier. Only trade
        # markets for days that haven't started yet.
        return None

    if target_hour is not None:
        # We tested this category live on 2026-09-07 and confirmed it's not
        # trustworthy: same-day "at a specific hour" markets settle only
        # hours away, and the market almost certainly already reflects the
        # ACTUAL current temperature (which anyone can just check), while
        # this only has a forecast made earlier that hasn't caught up.
        # Betting against a market that likely knows the real-time truth,
        # using stale data, is a good way to lose money -- so skip this
        # whole category rather than act on a signal we know is unreliable.
        return None
        # "At a specific hour" markets (e.g. KXTEMP*): every city's title
        # states the hour in EDT (UTC-4) regardless of the city's own local
        # timezone, so build the target instant in UTC and match hourly
        # forecast periods on that instant directly, rather than comparing
        # local hour numbers across timezones (which would silently
        # mismatch for Chicago/Austin).
        target_utc = dt.datetime(target_date.year, target_date.month, target_date.day,
                                  target_hour, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
        periods = _get_hourly_forecast_periods(*coords)
        if not periods:
            return None
        matching = [
            p for p in periods
            if dt.datetime.fromisoformat(p["startTime"]).astimezone(dt.timezone.utc)
               == target_utc.astimezone(dt.timezone.utc)
        ]
        if not matching:
            return None  # requested hour outside the hourly forecast horizon -- skip
        forecast_high = matching[0]["temperature"]
    else:
        # Whole-day "daily maximum" markets (e.g. KXHIGH*): use the daytime
        # period for that date from the standard (not hourly) forecast.
        periods = _get_forecast_periods(*coords)
        if not periods:
            return None
        matching = [
            p for p in periods
            if p.get("isDaytime") and dt.datetime.fromisoformat(p["startTime"]).date() == target_date
        ]
        if not matching:
            return None
        forecast_high = matching[0]["temperature"]

    if market.get("strike_type") != "greater":
        # Only "greater" strikes are handled for now (yes = temp above
        # floor_strike); other strike_types (e.g. the "-B##.5" between-style
        # tickers we saw) are skipped until their exact semantics are
        # confirmed, rather than guessed.
        return None
    threshold = market.get("floor_strike")
    if threshold is None:
        return None
    threshold = float(threshold)

    # Probability the actual high beats the threshold, modeled as a normal
    # distribution around the forecast with ~4.5F std dev (a rough estimate
    # of next-day NWS high-temp forecast error -- tune this against real
    # settlement outcomes over time, don't treat it as precise).
    from math import erf, sqrt
    std_dev = 4.5
    z = (forecast_high - threshold) / (std_dev * sqrt(2))
    prob_yes = 0.5 * (1 + erf(z))
    return max(0.02, min(0.98, prob_yes))


# --------------------------------------------------------------------------
# FINANCIALS -- same-day index-threshold markets vs. a live ETF-proxy
# quote, using a random-walk-with-volatility estimate of where the price
# lands by close. This is the same math behind option pricing (probability
# the underlying ends above a strike, given today's volatility and time
# remaining) -- a legitimate, independent estimate, not a fabricated one.
# --------------------------------------------------------------------------

_alpha_vantage_cache: dict = {}  # params tuple -> parsed response, this run only


def _alpha_vantage_get(params: dict) -> Optional[dict]:
    if not ALPHA_VANTAGE_API_KEY:
        return None
    cache_key = tuple(sorted(params.items()))
    if cache_key in _alpha_vantage_cache:
        return _alpha_vantage_cache[cache_key]
    try:
        resp = requests.get(ALPHA_VANTAGE_URL, params={**params, "apikey": ALPHA_VANTAGE_API_KEY}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning(f"Alpha Vantage request failed for {params}: {e}")
        return None
    # Alpha Vantage returns 200 OK with an "Error Message"/"Note"/
    # "Information" body for bad keys, rate limits, or bad symbols -- it
    # does NOT use HTTP error codes for these, so check the body itself.
    if not isinstance(data, dict) or any(k in data for k in ("Error Message", "Note", "Information")):
        log.warning(f"Alpha Vantage returned no usable data for {params}: {data}")
        return None
    _alpha_vantage_cache[cache_key] = data
    return data


def get_equity_quote(symbol: str) -> Optional[float]:
    """Latest price for symbol via Alpha Vantage GLOBAL_QUOTE. Gated to a
    few times a day -- see FINANCIAL_CHECK_HOURS_UTC -- to stay well under
    the free tier's 25-requests/day limit."""
    if dt.datetime.utcnow().hour not in FINANCIAL_CHECK_HOURS_UTC:
        return None
    data = _alpha_vantage_get({"function": "GLOBAL_QUOTE", "symbol": symbol})
    if not data:
        return None
    try:
        return float(data["Global Quote"]["05. price"])
    except (KeyError, TypeError, ValueError):
        return None


# Disclosed, round-number annualized volatility assumptions per proxy --
# NOT fitted from data. Treat this the same way as weather's "4.5F std
# dev": a rough starting point to be checked against real outcomes over
# time (via the prod-observation track record below), not a precise
# number. If this model earns its keep, replace with realized volatility
# computed from actual recent closes.
ANNUALIZED_VOL_ASSUMPTIONS = {"QQQ": 0.20, "SPY": 0.15}
TRADING_HOURS_PER_DAY = 6.5  # NYSE regular session, 9:30am-4:00pm ET


def financial_fair_value(market: dict) -> Optional[float]:
    """P(YES) for a same-day 'will <index> close above/below X today'
    Kalshi market, from a live ETF-proxy quote and an assumed volatility.
    Returns None (skip) for anything not confirmed to work: unknown
    series, unhandled strike_type, missing close time, a quote we
    couldn't get (rate-limit window closed or Alpha Vantage failed), or
    a market that has already closed."""
    event_ticker = market.get("event_ticker", "")
    proxy_symbol = next((sym for prefix, sym in FINANCIAL_SERIES.items() if event_ticker.startswith(prefix)), None)
    if not proxy_symbol:
        return None

    strike_type = market.get("strike_type")
    if strike_type not in ("greater", "less"):
        return None  # "between" markets skipped, same as weather -- semantics not yet confirmed
    threshold = market.get("floor_strike") if strike_type == "greater" else market.get("cap_strike")
    if threshold is None:
        return None

    close_time_raw = market.get("close_time") or market.get("expiration_time")
    if not close_time_raw:
        return None
    try:
        close_time = dt.datetime.fromisoformat(close_time_raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    hours_remaining = (close_time - dt.datetime.now(dt.timezone.utc)).total_seconds() / 3600
    if hours_remaining <= 0 or hours_remaining > 24:
        return None  # already closed, or this isn't actually a same-day market

    current_price = get_equity_quote(proxy_symbol)
    if current_price is None:
        return None

    annual_vol = ANNUALIZED_VOL_ASSUMPTIONS.get(proxy_symbol, 0.20)
    time_fraction = min(hours_remaining, TRADING_HOURS_PER_DAY) / TRADING_HOURS_PER_DAY
    daily_vol = annual_vol / (252 ** 0.5)
    sigma = current_price * daily_vol * (time_fraction ** 0.5)
    if sigma <= 0:
        return None

    from math import erf, sqrt
    z = (current_price - float(threshold)) / (sigma * sqrt(2))
    prob_above = 0.5 * (1 + erf(z))
    prob_yes = prob_above if strike_type == "greater" else (1 - prob_above)
    return max(0.02, min(0.98, prob_yes))


# --------------------------------------------------------------------------
# ECONOMICS -- CPI print-threshold markets vs. a trend extrapolation.
# WEAKER MODEL THAN WEATHER OR FINANCIALS, by design of what's available:
# Alpha Vantage's free tier gives actual past releases, not analyst
# consensus for the upcoming print. So this compares Kalshi's threshold
# to "recent trend continues," not "compare to an expert forecaster."
# Treat any edge this flags with real skepticism until it has a genuine
# settlement track record -- this is closer to an informed guess than
# either of the other two models.
# --------------------------------------------------------------------------

def get_recent_cpi_yoy_series() -> Optional[list]:
    """Last several monthly CPI YoY % values, oldest first. Gated to once
    a day -- CPI is a monthly release, there's no reason to ask more
    often, and it protects the shared 25/day Alpha Vantage budget."""
    if dt.datetime.utcnow().hour != ECONOMIC_CHECK_HOUR_UTC:
        return None
    data = _alpha_vantage_get({"function": "CPI", "interval": "monthly"})
    if not data:
        return None
    try:
        points = data["data"][:13]  # ~13 months, enough for a YoY comparison
        values = [float(p["value"]) for p in reversed(points)]  # oldest first
        if len(values) < 13:
            return None
        yoy = [(values[i] / values[i - 12] - 1) * 100 for i in range(12, len(values))]
        return yoy if yoy else None
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
        return None


def economic_fair_value(market: dict) -> Optional[float]:
    """P(YES) for a Kalshi CPI YoY threshold market, from a naive trend
    extrapolation (this month's YoY = same as the last reading) with a
    deliberately wide uncertainty band, reflecting how much weaker this
    signal is than a real consensus forecast. Only handles KXCPIYOY
    (year-over-year) for now -- KXCPI (month-over-month) needs a
    different Alpha Vantage series and isn't implemented, so it always
    returns None rather than guessing at the wrong statistic."""
    event_ticker = market.get("event_ticker", "")
    if not event_ticker.startswith("KXCPIYOY"):
        return None
    if market.get("strike_type") != "greater":
        return None
    threshold = market.get("floor_strike")
    if threshold is None:
        return None

    yoy_series = get_recent_cpi_yoy_series()
    if not yoy_series:
        return None
    trend_estimate = yoy_series[-1]  # naive: assume next print matches the last one

    from math import erf, sqrt
    std_dev = 0.35  # percentage points -- deliberately wide; this is a guess, not a forecast
    z = (trend_estimate - float(threshold)) / (std_dev * sqrt(2))
    prob_yes = 0.5 * (1 + erf(z))
    return max(0.05, min(0.95, prob_yes))  # tighter clamp than weather/financials -- reflects lower confidence


# --------------------------------------------------------------------------
# SPORTS -- Kalshi game-winner markets vs. no-vig consensus probability
# from real sportsbook moneylines. This is the classically legitimate
# version of this whole idea: comparing one book's price to an
# independent aggregate of others, the same thing professional sports
# bettors do. The tricky part isn't the math, it's matching a Kalshi
# event to the right Odds API event by date + teams.
# --------------------------------------------------------------------------

def american_odds_to_implied_prob(odds: float) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


_odds_api_cache: dict = {}  # sport_key -> list of events, this run only


def fetch_sportsbook_odds(sport_key: str) -> Optional[list]:
    if not ODDS_API_KEY:
        return None
    if sport_key in _odds_api_cache:
        return _odds_api_cache[sport_key]
    try:
        resp = requests.get(
            f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds",
            params={"apiKey": ODDS_API_KEY, "regions": "us", "markets": "h2h", "oddsFormat": "american"},
            timeout=10,
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        log.warning(f"Odds API request failed for {sport_key}: {e}")
        return None
    if not isinstance(events, list):
        log.warning(f"Odds API returned unexpected shape for {sport_key}: {events}")
        return None
    _odds_api_cache[sport_key] = events
    return events


def _no_vig_prob(team_name: str, event: dict) -> Optional[float]:
    """Average each bookmaker's no-vig (de-margined) probability for
    team_name, then average across bookmakers. Moneyline odds always
    imply >100% combined probability (the vig); dividing each side by
    that sum removes it before comparing books to each other."""
    probs = []
    for book in event.get("bookmakers", []):
        h2h = next((m for m in book.get("markets", []) if m.get("key") == "h2h"), None)
        if not h2h or len(h2h.get("outcomes", [])) != 2:
            continue
        outcomes = h2h["outcomes"]
        implied = {o["name"]: american_odds_to_implied_prob(o["price"]) for o in outcomes}
        total = sum(implied.values())
        if total <= 0 or team_name not in implied:
            continue
        probs.append(implied[team_name] / total)  # no-vig normalization
    if not probs:
        return None
    return sum(probs) / len(probs)


def sports_fair_value(market: dict) -> Optional[float]:
    """P(YES) for a Kalshi NFL game-winner market ('yes' = the team named
    in yes_sub_title/title wins), from the no-vig consensus across real
    sportsbooks. Returns None for anything not confidently matched --
    unknown series, a team-name/date pairing we can't map to an Odds API
    event, or no sportsbook data available for that game."""
    event_ticker = market.get("event_ticker", "")
    sport_key = next((sk for prefix, sk in SPORTS_SERIES.items() if event_ticker.startswith(prefix)), None)
    if not sport_key:
        return None

    # Kalshi's own city/team abbreviation for the side this specific
    # market resolves YES on is more reliable to parse from the ticker's
    # team-code suffix than from the free-text title. Ticker shape:
    # KXNFLGAME-{YY}{MON}{DD}{AWAY_CODE}{HOME_CODE}, e.g. ...-26SEP10SFLAR
    # (SF at LAR). Codes are 2-3 letters each with NO separator, so a
    # single regex can't tell where one ends and the other begins purely
    # from length -- greedy matching on a fixed split silently produces
    # the WRONG pair for some team combinations (confirmed while testing
    # this: "SFLAR" as [A-Z]{2,3}[A-Z]{2,3} greedily yields "SFL"+"AR",
    # neither a real team). Instead, extract the whole trailing code
    # block, then try every 2/3-length split and accept only the one
    # where BOTH halves are real codes from NFL_TEAM_CODES.
    import re
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})([A-Z]{4,6})$", event_ticker)
    if not m:
        return None
    yy, mon_abbr, dd, team_block = m.groups()
    try:
        month = dt.datetime.strptime(mon_abbr, "%b").month
        game_date = dt.date(2000 + int(yy), month, int(dd))
    except ValueError:
        return None

    valid_codes = set(NFL_TEAM_CODES.values())
    away_code = home_code = None
    for split in range(2, len(team_block) - 1):
        left, right = team_block[:split], team_block[split:]
        if left in valid_codes and right in valid_codes:
            away_code, home_code = left, right
            break
    if away_code is None:
        return None  # couldn't confidently split this ticker's team codes -- skip rather than guess

    events = fetch_sportsbook_odds(sport_key)
    if not events:
        return None

    yes_side_text = (market.get("yes_sub_title") or market.get("title") or "")
    candidate = None
    for event in events:
        try:
            commence = dt.datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if commence.date() not in (game_date, game_date + dt.timedelta(days=1)):
            continue  # allow +1 day slack for UTC vs. local-date game-day mismatches
        home, away = event.get("home_team"), event.get("away_team")
        if NFL_TEAM_CODES.get(home) == home_code and NFL_TEAM_CODES.get(away) == away_code:
            candidate = event
            break
    if candidate is None:
        return None

    home, away = candidate.get("home_team"), candidate.get("away_team")
    yes_team = next((t for t in (home, away) if t and t in yes_side_text), None)
    if yes_team is None:
        return None  # couldn't confidently tell which side this contract resolves YES on

    prob = _no_vig_prob(yes_team, candidate)
    if prob is None:
        return None
    return max(0.02, min(0.98, prob))


def estimate_fair_value(market: dict) -> Optional[float]:
    """Try every model we have real data for, in order. Each one already
    returns None for anything outside what it actually knows how to
    price, so trying them all here is safe -- at most one will ever
    return a non-None value for a given market, since they key off
    disjoint event_ticker prefixes."""
    for model in (weather_fair_value, financial_fair_value, economic_fair_value, sports_fair_value):
        result = model(market)
        if result is not None:
            return result
    return None


# --------------------------------------------------------------------------
# REAL-MARKET OBSERVATION (read-only, no credentials, never trades) --
# Demo markets carry no real volume or price discovery (confirmed live:
# volume_fp=0.00 and boundary prices of $0/$1 across the board), so they
# can never validate whether this model's edge is real. This function
# checks the SAME weather model against Kalshi's real, live, production
# prices instead -- using Kalshi's public, unauthenticated market-data
# endpoints, which need no API key and no funded account. It only reads
# data and logs what it finds; it never places an order and never uses
# LIVE_TRADING. This runs every tick regardless of KALSHI_ENV, so a track
# record builds up that can later be checked against actual settlement
# outcomes (Kalshi's own public API reports each market's final "result"
# once it resolves) -- BEFORE ever trusting this signal with real money.
# --------------------------------------------------------------------------

def fetch_prod_markets_public(series_ticker: str) -> list:
    """Unauthenticated GET against Kalshi's real production servers. No
    KalshiClient, no signing, no credentials -- GetMarkets is public."""
    markets, cursor = [], None
    while True:
        params = {"series_ticker": series_ticker, "status": "open", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        try:
            resp = requests.get(f"{PROD_BASE_URL}/markets", params=params, timeout=10)
            resp.raise_for_status()
            page = resp.json()
        except Exception as e:
            log.warning(f"prod market fetch failed for {series_ticker}: {e}")
            return markets
        markets.extend(page.get("markets", []))
        cursor = page.get("cursor")
        if not cursor:
            break
    return markets


# Every category we have a model for, so both the demo trading loop and
# the real-prod observation pass can walk the same list instead of
# duplicating it. "category" here is just a label for the ledger/logs --
# estimate_fair_value() still dispatches to the right model on its own
# via event_ticker prefix, so nothing here needs to key models to series
# manually or risk mismatching one.
ALL_SERIES_REGISTRIES = {
    "weather": NWS_GRIDPOINTS,
    "financial": FINANCIAL_SERIES,
    "economic": ECONOMIC_SERIES,
    "sports": SPORTS_SERIES,
}


def observe_prod_edges() -> list:
    """One read-only pass over every real Kalshi market we have a model
    for, across all categories. Returns a list of observation dicts
    (empty list on total failure -- never raises, so a Kalshi/Alpha
    Vantage/Odds API hiccup here can't take down the actual trading
    tick)."""
    observations = []
    for category, series_registry in ALL_SERIES_REGISTRIES.items():
        for series_ticker in series_registry:
            try:
                markets = fetch_prod_markets_public(series_ticker)
            except Exception as e:
                log.warning(f"prod observation skipped for {series_ticker}: {e}")
                continue
            for market in markets:
                try:
                    fair_prob = estimate_fair_value(market)
                    if fair_prob is None:
                        continue
                    yes_ask = _dollars(market, "yes_ask_dollars")
                    yes_bid = _dollars(market, "yes_bid_dollars")
                    if yes_ask <= 0 or yes_ask >= 1:
                        continue  # same "not a real tradeable price" guard evaluate_market uses
                    observations.append({
                        "t": dt.datetime.utcnow().isoformat() + "Z",
                        "category": category,
                        "ticker": market.get("ticker", ""),
                        "title": market.get("title", market.get("ticker", "")),
                        "event_ticker": market.get("event_ticker", ""),
                        "fair_prob": round(fair_prob, 3),
                        "market_price": round(yes_ask, 3),
                        "market_bid": round(yes_bid, 3),
                        "edge": round(fair_prob - yes_ask, 3),
                        "volume_fp": market.get("volume_fp"),
                    })
                except Exception as e:
                    log.warning(f"prod observation skipped for market {market.get('ticker')}: {e}")
    return observations


# --------------------------------------------------------------------------
# SIZING -- Kelly criterion, capped, exactly like the video described.
# --------------------------------------------------------------------------

def kelly_fraction(fair_prob: float, market_price: float) -> float:
    """Binary-market Kelly fraction for buying YES at market_price (0-1)
    when you believe the true probability is fair_prob."""
    if market_price <= 0 or market_price >= 1:
        return 0.0
    b = (1 - market_price) / market_price  # net odds
    f = (fair_prob * (b + 1) - 1) / b
    return max(0.0, f)


@dataclass
class Decision:
    ticker: str
    title: str
    side: str
    fair_prob: float
    market_price: float
    edge: float
    kelly_frac: float
    dollars: int
    contracts: int


def _dollars(market: dict, key: str, default: float = 0.0) -> float:
    """Kalshi's current schema returns prices as strings like '0.0100'
    under keys ending in _dollars, not integer cents under bare names."""
    raw = market.get(key)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def evaluate_market(market: dict, bankroll_cents: int) -> tuple:
    """Returns (Decision_or_None, diagnostic). diagnostic always carries a
    "stage" describing how far this market got, and an "edge" (float or
    None) whenever an edge was actually computable -- this is what lets a
    tick that places zero orders still explain *how close* it got, instead
    of just "0 live orders placed" with no visibility into why."""
    if _dollars(market, "volume_fp") < EFFECTIVE_MIN_VOLUME:
        return None, {"stage": "too_thin", "edge": None}

    fair_prob = estimate_fair_value(market)
    if fair_prob is None:
        return None, {"stage": "no_model", "edge": None}

    yes_ask_dollars = _dollars(market, "yes_ask_dollars")
    yes_ask = int(round(yes_ask_dollars * 100))  # convert to whole cents
    if yes_ask <= 0 or yes_ask >= 100:
        return None, {"stage": "no_price", "edge": None}
    market_price = yes_ask / 100.0

    edge = fair_prob - market_price
    side = "yes" if edge > 0 else "no"
    effective_price = market_price if side == "yes" else (1 - market_price)
    effective_fair = fair_prob if side == "yes" else (1 - fair_prob)

    if abs(edge) < EDGE_THRESHOLD:
        return None, {"stage": "below_threshold", "edge": edge}

    frac = kelly_fraction(effective_fair, effective_price) * KELLY_MULTIPLIER
    frac = min(frac, MAX_KELLY_FRACTION)
    if frac <= 0:
        return None, {"stage": "kelly_zero", "edge": edge}

    dollars = int(bankroll_cents * frac)
    price_cents = yes_ask if side == "yes" else (100 - yes_ask)
    contracts = max(0, dollars // max(price_cents, 1))
    if contracts < 1:
        return None, {"stage": "sized_to_zero_contracts", "edge": edge}

    title = market.get("title") or market["ticker"]
    decision = Decision(market["ticker"], title, side, fair_prob, market_price, edge, frac, dollars, contracts)
    return decision, {"stage": "executed_or_paper", "edge": edge}


# --------------------------------------------------------------------------
# LEDGER -- simple CSV log so you can build your own version of the
# "dashboard" from the video. Point this at a real DB (Supabase, etc.) if
# you deploy this as a Render cron job, since local disk won't persist
# between runs.
# --------------------------------------------------------------------------

def log_decision(d: Decision, executed: bool, note: str = ""):
    is_new = not LEDGER_PATH.exists()
    with open(LEDGER_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "ticker", "side", "fair_prob", "market_price",
                        "edge", "kelly_frac", "dollars", "contracts", "executed", "note"])
        w.writerow([dt.datetime.utcnow().isoformat(), d.ticker, d.side, round(d.fair_prob, 3),
                    round(d.market_price, 3), round(d.edge, 3), round(d.kelly_frac, 3),
                    d.dollars, d.contracts, executed, note])


def today_realized_pnl_pct(bankroll_cents: int) -> float:
    """Best-effort daily-loss check from the ledger. Returns 0.0 if it
    can't tell (e.g. first run of the day) -- extend with real fills data
    from /portfolio/fills for something more precise."""
    if not LEDGER_PATH.exists():
        return 0.0
    today = dt.date.today().isoformat()
    spent = 0
    with open(LEDGER_PATH) as f:
        for row in csv.DictReader(f):
            if row["timestamp"].startswith(today) and row["executed"] == "True":
                spent += int(row["dollars"])
    return -spent / max(bankroll_cents, 1)


# --------------------------------------------------------------------------
# PUBLIC GITHUB LEDGER (optional) -- lets a free static dashboard read the
# bot's activity with zero backend and zero database. This is meant purely
# as a data drop: it should live in its own small PUBLIC repo containing
# nothing but this one JSON file. Never point GITHUB_LEDGER_REPO at the
# private strategy repo, and never give this token more than "Contents:
# Read and write" on that one repo.
# --------------------------------------------------------------------------

def _github_ledger_configured() -> bool:
    return bool(GITHUB_LEDGER_TOKEN and GITHUB_LEDGER_REPO and GITHUB_LEDGER_PATH)


def _github_get_ledger() -> tuple:
    """Returns (ledger_dict, sha_or_None). sha is None if the file doesn't
    exist yet (first run) -- GitHub requires the current sha to update an
    existing file, but rejects a sha on first creation."""
    url = f"https://api.github.com/repos/{GITHUB_LEDGER_REPO}/contents/{GITHUB_LEDGER_PATH}"
    headers = {
        "Authorization": f"Bearer {GITHUB_LEDGER_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    resp = requests.get(url, headers=headers, timeout=10)
    if resp.status_code == 404:
        return {"bankroll_history": [], "decisions": []}, None
    resp.raise_for_status()
    payload = resp.json()
    import json
    content = base64.b64decode(payload["content"]).decode("utf-8")
    return json.loads(content), payload["sha"]


def _github_put_ledger(ledger: dict, sha: Optional[str]):
    import json
    url = f"https://api.github.com/repos/{GITHUB_LEDGER_REPO}/contents/{GITHUB_LEDGER_PATH}"
    headers = {
        "Authorization": f"Bearer {GITHUB_LEDGER_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body = {
        "message": f"ledger update {dt.datetime.utcnow().isoformat()}",
        "content": base64.b64encode(json.dumps(ledger, indent=2).encode("utf-8")).decode("utf-8"),
    }
    if sha:
        body["sha"] = sha
    resp = requests.put(url, json=body, headers=headers, timeout=10)
    resp.raise_for_status()


def push_ledger_to_github(bankroll_cents: int, tick_decisions: list, prod_observations: Optional[list] = None):
    """Appends this tick's bankroll snapshot, any decisions, and any real
    prod-market edge observations to the public ledger repo. Best-effort:
    any failure here is logged and swallowed -- a GitHub hiccup should
    never take down the actual trading loop."""
    if not _github_ledger_configured():
        return
    try:
        ledger, sha = _github_get_ledger()
        ledger.setdefault("bankroll_history", []).append({
            "t": dt.datetime.utcnow().isoformat() + "Z",
            "dollars": round(bankroll_cents / 100, 2),
        })
        ledger["bankroll_history"] = ledger["bankroll_history"][-MAX_LEDGER_BANKROLL_POINTS:]

        ledger.setdefault("decisions", [])
        ledger["decisions"].extend(tick_decisions)
        ledger["decisions"] = ledger["decisions"][-MAX_LEDGER_DECISIONS:]

        if prod_observations:
            ledger.setdefault("prod_edge_observations", [])
            ledger["prod_edge_observations"].extend(prod_observations)
            ledger["prod_edge_observations"] = ledger["prod_edge_observations"][-MAX_PROD_OBSERVATIONS:]

        ledger["updated_at"] = dt.datetime.utcnow().isoformat() + "Z"
        ledger["env"] = KALSHI_ENV

        _github_put_ledger(ledger, sha)
        log.info(
            f"pushed ledger update to {GITHUB_LEDGER_REPO} "
            f"({len(tick_decisions)} new decisions, "
            f"{len(prod_observations or [])} new prod observations)"
        )
    except Exception as e:
        log.warning(f"github ledger push failed (non-fatal): {e}")


# --------------------------------------------------------------------------
# MAIN -- one tick of the loop. Call this every 10 minutes from a scheduler.
# --------------------------------------------------------------------------

def main():
    client = KalshiClient(KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH, BASE_URL)

    bankroll_cents = client.get_balance_cents()
    log.info(f"[{KALSHI_ENV}] bankroll: ${bankroll_cents/100:.2f}")
    if KALSHI_ENV != "prod":
        log.info(
            f"volume floor disabled for {KALSHI_ENV} (demo markets carry no real "
            f"liquidity signal) -- MIN_CONTRACT_VOLUME={MIN_CONTRACT_VOLUME} is NOT "
            f"being enforced this run; it activates automatically in prod"
        )

    if today_realized_pnl_pct(bankroll_cents) < -MAX_DAILY_LOSS_PCT:
        log.warning("Daily loss cap hit -- skipping this tick.")
        return

    markets = []
    for category, series_registry in ALL_SERIES_REGISTRIES.items():
        category_markets = client.get_known_markets(series_registry)
        log.info(f"pulled {len(category_markets)} markets from known {category} series")
        markets.extend(category_markets)

    trades_this_tick = 0
    tick_decisions = []  # for the public GitHub ledger, if configured
    tick_started_at = dt.datetime.utcnow().isoformat() + "Z"

    edge_computed_count = 0
    best_near_miss = None  # (abs_edge, ticker, title, edge) -- for the "how close did we get" summary
    stage_counts: dict = {}  # tally of diag["stage"] across the tick -- pinpoints WHERE markets get filtered out

    for i, market in enumerate(markets):
        decision, diag = evaluate_market(market, bankroll_cents)
        stage_counts[diag["stage"]] = stage_counts.get(diag["stage"], 0) + 1

        if i < 3:
            # Raw sample of what Kalshi is actually returning, once per tick,
            # so a "why is everything filtered out" question can be answered
            # by looking at real field values instead of guessing.
            log.info(
                f"sample market: ticker={market.get('ticker')} "
                f"event_ticker={market.get('event_ticker')} "
                f"volume_fp={market.get('volume_fp')} "
                f"floor_strike={market.get('floor_strike')} "
                f"strike_type={market.get('strike_type')} "
                f"yes_ask_dollars={market.get('yes_ask_dollars')} "
                f"-> stage={diag['stage']}"
            )

        if diag.get("edge") is not None:
            edge_computed_count += 1
            edge = diag["edge"]
            if best_near_miss is None or abs(edge) > best_near_miss[0]:
                best_near_miss = (abs(edge), market["ticker"], market.get("title") or market["ticker"], edge)

        if decision is None:
            continue

        log.info(
            f"EDGE FOUND: {decision.ticker} side={decision.side} "
            f"fair={decision.fair_prob:.2f} price={decision.market_price:.2f} "
            f"edge={decision.edge:+.2f} size=${decision.dollars/100:.2f} "
            f"({decision.contracts} contracts)"
        )

        status, note = "paper", "paper mode"
        if LIVE_TRADING:
            # NOTE: order-placement field names (yes_price/no_price as cents
            # ints in place_order()) come from Kalshi's documented order
            # lifecycle guide, but haven't been confirmed live the way the
            # market-reading fields just were -- verify against a real order
            # preview/response before trusting this on real money.
            yes_ask_cents = int(round(_dollars(market, "yes_ask_dollars") * 100))
            price_cents = yes_ask_cents if decision.side == "yes" else (100 - yes_ask_cents)
            try:
                client.place_order(decision.ticker, decision.side, decision.contracts, price_cents)
                log_decision(decision, executed=True)
                trades_this_tick += 1
                status, note = "executed", ""
            except Exception as e:
                log.error(f"order failed for {decision.ticker}: {e}")
                log_decision(decision, executed=False, note=str(e))
                status, note = "failed", str(e)
        else:
            log_decision(decision, executed=False, note="paper mode")

        tick_decisions.append({
            "t": tick_started_at,
            "ticker": decision.ticker,
            "title": decision.title,
            "side": decision.side,
            "fair_prob": round(decision.fair_prob, 3),
            "market_price": round(decision.market_price, 3),
            "edge": round(decision.edge, 3),
            "dollars": round(decision.dollars / 100, 2),
            "contracts": decision.contracts,
            "status": status,
            "note": note,
        })

    log.info(f"tick complete -- {trades_this_tick} live orders placed")
    log.info(f"filter stage breakdown this tick: {stage_counts}")
    if best_near_miss:
        _, ticker, title, edge = best_near_miss
        log.info(
            f"closest miss this tick: {ticker} ({title}) edge={edge:+.3f} "
            f"vs threshold {EDGE_THRESHOLD:.2f} -- {edge_computed_count}/{len(markets)} "
            f"markets had a computable edge (rest were same-day, too thin, or not a "
            f"weather-modeled ticker)"
        )
    else:
        log.info(
            f"no markets had a computable edge this tick ({len(markets)} scanned) -- "
            f"likely all same-day, or below the {EFFECTIVE_MIN_VOLUME}-contract volume "
            f"floor (see 'filter stage breakdown' above for the exact reason)"
        )

    # Read-only check against REAL prod prices, regardless of KALSHI_ENV --
    # never places an order, never needs credentials. This is how we build
    # a track record to validate against actual settlement outcomes before
    # ever trusting this signal with real money (see the note above
    # observe_prod_edges for why demo data can't do this).
    prod_observations = []
    try:
        prod_observations = observe_prod_edges()
        if prod_observations:
            biggest = max(prod_observations, key=lambda o: abs(o["edge"]))
            log.info(
                f"prod check: {len(prod_observations)} real market(s) with a "
                f"computable edge; largest={biggest['edge']:+.3f} on "
                f"{biggest['ticker']} ({biggest['title']}) -- fair={biggest['fair_prob']:.3f} "
                f"vs real market price={biggest['market_price']:.3f}, "
                f"volume={biggest['volume_fp']}"
            )
        else:
            log.info("prod check: no real markets had a computable edge this tick")
    except Exception as e:
        log.warning(f"prod edge observation failed (non-fatal): {e}")

    push_ledger_to_github(bankroll_cents, tick_decisions, prod_observations)


if __name__ == "__main__":
    main()
