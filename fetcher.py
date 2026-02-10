"""
Polymarket API data fetcher.
Connects to CLOB API (prices, orderbooks) and Gamma API (market discovery).
"""
import time
import logging
import requests

from config import CLOB_BASE_URL, GAMMA_BASE_URL, API_DELAY, MARKETS_PER_PAGE
from models import Market, Condition, Token

logger = logging.getLogger(__name__)


class PolymarketFetcher:
    """Fetches market data from Polymarket APIs (no auth needed for read-only)."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------ #
    #  Gamma API - Market Discovery
    # ------------------------------------------------------------------ #

    def get_events(self, limit: int = 100, offset: int = 0, **filters) -> list[dict]:
        """Fetch events from Gamma API (Section 4.1)."""
        params = {"limit": limit, "offset": offset, **filters}
        resp = self._get(f"{GAMMA_BASE_URL}/events", params=params)
        return resp if isinstance(resp, list) else []

    def get_markets_gamma(self, limit: int = 100, offset: int = 0, **filters) -> list[dict]:
        """Fetch individual markets from Gamma API."""
        params = {"limit": limit, "offset": offset, **filters}
        resp = self._get(f"{GAMMA_BASE_URL}/markets", params=params)
        return resp if isinstance(resp, list) else []

    def get_all_active_events(self) -> list[dict]:
        """Paginate through all active, non-closed events."""
        all_events = []
        offset = 0
        while True:
            batch = self.get_events(
                limit=MARKETS_PER_PAGE,
                offset=offset,
                active="true",
                closed="false",
            )
            if not batch:
                break
            all_events.extend(batch)
            offset += len(batch)
            logger.info(f"Fetched {len(all_events)} events...")
            time.sleep(API_DELAY)
        return all_events

    # ------------------------------------------------------------------ #
    #  CLOB API - Prices and Order Books
    # ------------------------------------------------------------------ #

    def get_price(self, token_id: str, side: str = "buy") -> float | None:
        """Get current price for a token (Section 4.2)."""
        resp = self._get(
            f"{CLOB_BASE_URL}/price",
            params={"token_id": token_id, "side": side},
        )
        if resp and "price" in resp:
            return float(resp["price"])
        return None

    def get_midpoint(self, token_id: str) -> float | None:
        """Get midpoint price for a token."""
        resp = self._get(
            f"{CLOB_BASE_URL}/midpoint",
            params={"token_id": token_id},
        )
        if resp and "mid" in resp:
            return float(resp["mid"])
        return None

    def get_order_book(self, token_id: str) -> dict | None:
        """Get full order book for a token."""
        return self._get(
            f"{CLOB_BASE_URL}/book",
            params={"token_id": token_id},
        )

    def get_clob_markets(self, next_cursor: str = "") -> dict:
        """Get markets from CLOB with pagination."""
        params = {}
        if next_cursor:
            params["next_cursor"] = next_cursor
        return self._get(f"{CLOB_BASE_URL}/markets", params=params) or {}

    # ------------------------------------------------------------------ #
    #  Parsing - Build Market/Condition objects from API responses
    # ------------------------------------------------------------------ #

    def parse_event_to_markets(self, event: dict) -> list[Market]:
        """
        Parse a Gamma API event into Market objects.
        Each event can have multiple markets (conditions in NegRisk events).
        """
        markets_data = event.get("markets", [])
        if not markets_data:
            return []

        neg_risk = event.get("negRisk", False)

        if neg_risk:
            # NegRisk event: all sub-markets are conditions of one logical market
            market = Market(
                market_id=event.get("id", ""),
                slug=event.get("slug", ""),
                question=event.get("title", ""),
                description=event.get("description", ""),
                neg_risk=True,
                end_date=event.get("endDate", ""),
                volume=float(event.get("volume", 0) or 0),
                liquidity=float(event.get("liquidity", 0) or 0),
                active=event.get("active", True),
            )
            for md in markets_data:
                condition = self._parse_condition(md)
                if condition:
                    market.conditions.append(condition)
            return [market] if market.conditions else []
        else:
            # Independent single-condition markets
            result = []
            for md in markets_data:
                condition = self._parse_condition(md)
                if not condition:
                    continue
                m = Market(
                    market_id=md.get("id", ""),
                    slug=md.get("slug", event.get("slug", "")),
                    question=md.get("question", ""),
                    description=md.get("description", ""),
                    neg_risk=False,
                    end_date=md.get("endDateIso", ""),
                    volume=float(md.get("volume", 0) or 0),
                    liquidity=float(md.get("liquidity", 0) or 0),
                    active=md.get("active", True),
                    conditions=[condition],
                )
                result.append(m)
            return result

    def _parse_condition(self, market_data: dict) -> Condition | None:
        """Parse a single market entry into a Condition with tokens."""
        clob_token_ids = market_data.get("clobTokenIds", [])
        outcomes = market_data.get("outcomes", [])
        outcome_prices = market_data.get("outcomePrices", [])

        if not clob_token_ids or not outcomes:
            return None

        tokens = []
        for i, outcome in enumerate(outcomes):
            price = 0.0
            if i < len(outcome_prices) and outcome_prices[i]:
                try:
                    price = float(outcome_prices[i])
                except (ValueError, TypeError):
                    pass
            token_id = clob_token_ids[i] if i < len(clob_token_ids) else ""
            tokens.append(Token(token_id=token_id, outcome=outcome, price=price))

        return Condition(
            condition_id=market_data.get("conditionId", ""),
            question=market_data.get("question", ""),
            tokens=tokens,
        )

    def refresh_prices(self, market: Market) -> Market:
        """Refresh live prices from CLOB for all tokens in a market."""
        for condition in market.conditions:
            for token in condition.tokens:
                if not token.token_id:
                    continue
                mid = self.get_midpoint(token.token_id)
                if mid is not None:
                    token.price = mid
                time.sleep(API_DELAY)
        return market

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _get(self, url: str, params: dict = None) -> dict | list | None:
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"API request failed: {url} - {e}")
            return None
