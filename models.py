"""
Data models for Polymarket markets and arbitrage opportunities.
"""
from dataclasses import dataclass, field
from enum import Enum


class ArbitrageType(Enum):
    """Types of arbitrage as defined in the paper (Section 3.2)."""
    SINGLE_CONDITION = "single_condition"       # YES + NO != 1 within one condition
    MARKET_REBALANCING_LONG = "rebalancing_long"  # Sum(YES) < 1 across conditions
    MARKET_REBALANCING_SHORT = "rebalancing_short" # Sum(YES) > 1 across conditions
    COMBINATORIAL = "combinatorial"              # Between dependent markets


@dataclass
class Token:
    token_id: str
    outcome: str  # "Yes" or "No"
    price: float


@dataclass
class Condition:
    """A single condition (binary outcome) within a market."""
    condition_id: str
    question: str
    tokens: list[Token] = field(default_factory=list)

    @property
    def yes_price(self) -> float:
        for t in self.tokens:
            if t.outcome.lower() == "yes":
                return t.price
        return 0.0

    @property
    def no_price(self) -> float:
        for t in self.tokens:
            if t.outcome.lower() == "no":
                return t.price
        return 0.0

    @property
    def spread(self) -> float:
        """Deviation from $1. Negative = long arb, positive = short arb."""
        return (self.yes_price + self.no_price) - 1.0


@dataclass
class Market:
    """A Polymarket market (event with one or more conditions)."""
    market_id: str
    slug: str
    question: str
    description: str
    conditions: list[Condition] = field(default_factory=list)
    neg_risk: bool = False
    end_date: str = ""
    volume: float = 0.0
    liquidity: float = 0.0
    active: bool = True

    @property
    def is_multi_condition(self) -> bool:
        return len(self.conditions) > 1 or self.neg_risk

    @property
    def yes_price_sum(self) -> float:
        """Sum of all YES prices across conditions (Def. 3)."""
        return sum(c.yes_price for c in self.conditions)


@dataclass
class ArbitrageOpportunity:
    """A detected arbitrage opportunity."""
    arb_type: ArbitrageType
    market: Market
    profit_per_dollar: float  # |1 - sum(YES)| or |sum_S - sum_S'|
    details: str
    conditions_involved: list[Condition] = field(default_factory=list)
    second_market: Market | None = None  # For combinatorial arbitrage

    @property
    def strategy(self) -> str:
        if self.arb_type == ArbitrageType.SINGLE_CONDITION:
            if self.profit_per_dollar > 0:
                return "BUY YES + NO (sum < $1)"
            return "SPLIT & SELL YES + NO (sum > $1)"
        elif self.arb_type == ArbitrageType.MARKET_REBALANCING_LONG:
            return "BUY all YES positions (sum < $1)"
        elif self.arb_type == ArbitrageType.MARKET_REBALANCING_SHORT:
            return "BUY all NO positions (sum > $1)"
        else:
            return "Cross-market positions"

    @property
    def estimated_max_profit_usd(self) -> float:
        """Estimated profit based on available liquidity."""
        from config import MAX_POSITION_SIZE
        return abs(self.profit_per_dollar) * min(self.market.liquidity, MAX_POSITION_SIZE)
