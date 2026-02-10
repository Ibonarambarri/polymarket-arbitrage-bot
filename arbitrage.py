"""
Arbitrage detection engine.

Implements the two types of arbitrage from the paper:
  1. Market Rebalancing Arbitrage (Definition 3) - within a single market
  2. Combinatorial Arbitrage (Definition 4) - between dependent markets

Reference: "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"
           Saguillo, Ghafouri, Kiffer, Suarez-Tangil (2025)
"""
from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from config import MIN_PROFIT_MARGIN, MAX_CONDITION_PRICE, EMBEDDING_SIMILARITY_THRESHOLD
from models import (
    Market,
    Condition,
    ArbitrageOpportunity,
    ArbitrageType,
)

if TYPE_CHECKING:
    from llm_detector import LLMDependencyDetector
    from embeddings import TopicClassifier

logger = logging.getLogger(__name__)


class ArbitrageDetector:
    """Scans markets for arbitrage opportunities."""

    def __init__(
        self,
        min_margin: float = MIN_PROFIT_MARGIN,
        llm_detector: LLMDependencyDetector | None = None,
        topic_classifier: TopicClassifier | None = None,
    ):
        self.min_margin = min_margin
        self.llm_detector = llm_detector
        self.topic_classifier = topic_classifier
        self._checked_pairs: set[tuple[str, str]] = set()

    # ================================================================== #
    #  1. Single Condition Arbitrage (Section 6.1)
    # ================================================================== #

    def check_single_condition(self, market: Market) -> list[ArbitrageOpportunity]:
        """
        For each condition, check if YES + NO prices deviate from $1.
        If YES + NO < 1: buy both -> guaranteed profit when market resolves.
        If YES + NO > 1: split & sell both -> instant profit.
        """
        opportunities = []
        for condition in market.conditions:
            if not self._is_uncertain(condition):
                continue

            spread = condition.yes_price + condition.no_price
            deviation = abs(spread - 1.0)

            if deviation < self.min_margin:
                continue

            if spread < 1.0:
                profit = 1.0 - spread
                detail = (
                    f"YES={condition.yes_price:.4f} + NO={condition.no_price:.4f} "
                    f"= {spread:.4f} < $1.00 | Buy both -> profit ${profit:.4f}/unit"
                )
            else:
                profit = spread - 1.0
                detail = (
                    f"YES={condition.yes_price:.4f} + NO={condition.no_price:.4f} "
                    f"= {spread:.4f} > $1.00 | Split & sell -> profit ${profit:.4f}/unit"
                )

            opportunities.append(ArbitrageOpportunity(
                arb_type=ArbitrageType.SINGLE_CONDITION,
                market=market,
                profit_per_dollar=profit,
                details=detail,
                conditions_involved=[condition],
            ))

        return opportunities

    # ================================================================== #
    #  2. Market Rebalancing Arbitrage (Definition 3, Section 6.2)
    # ================================================================== #

    def check_market_rebalancing(self, market: Market) -> list[ArbitrageOpportunity]:
        """
        For NegRisk markets with multiple conditions:
        Check if sum of all YES prices deviates from $1.

        Long arb:  sum(YES) < 1 -> buy all YES -> profit = 1 - sum
        Short arb: sum(YES) > 1 -> buy all NO  -> profit = sum - 1
        """
        if not market.is_multi_condition or len(market.conditions) < 2:
            return []

        # Filter to uncertain conditions (paper: ignore conditions > 95%)
        active_conditions = [c for c in market.conditions if self._is_uncertain(c)]
        if len(active_conditions) < 2:
            return []

        yes_sum = sum(c.yes_price for c in active_conditions)
        deviation = abs(yes_sum - 1.0)

        if deviation < self.min_margin:
            return []

        if yes_sum < 1.0:
            profit = 1.0 - yes_sum
            prices_str = " + ".join(f"{c.yes_price:.3f}" for c in active_conditions)
            return [ArbitrageOpportunity(
                arb_type=ArbitrageType.MARKET_REBALANCING_LONG,
                market=market,
                profit_per_dollar=profit,
                details=(
                    f"LONG: sum(YES) = {prices_str} = {yes_sum:.4f} < $1.00\n"
                    f"  Buy 1 YES of each condition -> guaranteed profit ${profit:.4f}/unit\n"
                    f"  Conditions: {len(active_conditions)}"
                ),
                conditions_involved=active_conditions,
            )]
        else:
            profit = yes_sum - 1.0
            prices_str = " + ".join(f"{c.yes_price:.3f}" for c in active_conditions)
            return [ArbitrageOpportunity(
                arb_type=ArbitrageType.MARKET_REBALANCING_SHORT,
                market=market,
                profit_per_dollar=profit,
                details=(
                    f"SHORT: sum(YES) = {prices_str} = {yes_sum:.4f} > $1.00\n"
                    f"  Buy 1 NO of each condition -> guaranteed profit ${profit:.4f}/unit\n"
                    f"  Conditions: {len(active_conditions)}"
                ),
                conditions_involved=active_conditions,
            )]

    # ================================================================== #
    #  3. Combinatorial Arbitrage (Definition 4, Section 6.3)
    # ================================================================== #

    def find_dependent_pairs(
        self, markets: list[Market], similarity_threshold: float = 0.5
    ) -> list[tuple[Market, Market, list[tuple[Condition, Condition]]]]:
        """
        Find pairs of markets with dependent conditions.

        Three modes (from most to least accurate):
        1. LLM-based (paper Section 5): logical dependency via LLM reasoning
        2. Embedding-based: topic + semantic similarity pre-filtering
        3. Heuristic fallback: text similarity with SequenceMatcher
        """
        pairs = []
        self._checked_pairs.clear()

        # Group markets by (topic, date) or just by date
        if self.topic_classifier:
            groups = self.topic_classifier.group_by_topic_and_date(markets)
        else:
            groups = self._group_by_date(markets)

        for group_key, group in groups.items():
            if len(group) < 2:
                continue

            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    m1, m2 = group[i], group[j]
                    if m1.market_id == m2.market_id:
                        continue

                    pair_key = tuple(sorted([m1.market_id, m2.market_id]))
                    if pair_key in self._checked_pairs:
                        continue
                    self._checked_pairs.add(pair_key)

                    # Pre-filter: embedding similarity (if available)
                    if self.topic_classifier:
                        sim = self.topic_classifier.compute_similarity(m1, m2)
                        if sim < EMBEDDING_SIMILARITY_THRESHOLD:
                            continue

                    # Check dependency: LLM or heuristic
                    dep_conditions = self._check_pair_dependency(
                        m1, m2, similarity_threshold
                    )
                    if dep_conditions:
                        pairs.append((m1, m2, dep_conditions))

        logger.info(f"Found {len(pairs)} dependent market pairs")
        return pairs

    def _check_pair_dependency(
        self,
        m1: Market,
        m2: Market,
        similarity_threshold: float,
    ) -> list[tuple[Condition, Condition]]:
        """
        Check if two markets are dependent using LLM (if available)
        or fallback to heuristic text matching.
        """
        # Mode 1: LLM-based dependency detection (paper Section 5.2)
        if self.llm_detector:
            result = self.llm_detector.check_market_pair(m1, m2)
            if result.is_dependent:
                return self._extract_conditions_from_llm(result, m1, m2)
            return []

        # Mode 2: Heuristic fallback
        sim = self._text_similarity(m1.question, m2.question)
        if sim < similarity_threshold:
            return []
        return self._find_dependent_conditions_heuristic(m1, m2)

    def _extract_conditions_from_llm(
        self, result, m1: Market, m2: Market
    ) -> list[tuple[Condition, Condition]]:
        """Extract dependent condition pairs from LLM dependency result."""
        deps = []
        for m1_indices, m2_indices in result.dependent_subsets:
            conds_m1 = self.llm_detector._reduce_conditions(m1)
            conds_m2 = self.llm_detector._reduce_conditions(m2)
            for i in m1_indices:
                if i < len(conds_m1):
                    for j in m2_indices:
                        if j < len(conds_m2):
                            deps.append((conds_m1[i], conds_m2[j]))
        return deps

    def check_combinatorial_arbitrage(
        self,
        market1: Market,
        market2: Market,
        dependent_conditions: list[tuple[Condition, Condition]],
    ) -> list[ArbitrageOpportunity]:
        """
        Check for combinatorial arbitrage between dependent markets (Def. 4).

        For dependent subsets S in M1 and S' in M2:
        If sum(S_YES) != sum(S'_YES), arbitrage exists.
        """
        opportunities = []

        for c1, c2 in dependent_conditions:
            if not self._is_uncertain(c1) or not self._is_uncertain(c2):
                continue

            price_diff = abs(c1.yes_price - c2.yes_price)
            if price_diff < self.min_margin:
                continue

            if c1.yes_price < c2.yes_price:
                detail = (
                    f"M1 '{c1.question}' YES={c1.yes_price:.4f} vs "
                    f"M2 '{c2.question}' YES={c2.yes_price:.4f}\n"
                    f"  Buy YES in M1, buy NO in M2 -> profit ${price_diff:.4f}/unit"
                )
            else:
                detail = (
                    f"M1 '{c1.question}' YES={c1.yes_price:.4f} vs "
                    f"M2 '{c2.question}' YES={c2.yes_price:.4f}\n"
                    f"  Buy NO in M1, buy YES in M2 -> profit ${price_diff:.4f}/unit"
                )

            opportunities.append(ArbitrageOpportunity(
                arb_type=ArbitrageType.COMBINATORIAL,
                market=market1,
                second_market=market2,
                profit_per_dollar=price_diff,
                details=detail,
                conditions_involved=[c1, c2],
            ))

        return opportunities

    # ================================================================== #
    #  Full Scan
    # ================================================================== #

    def scan_all(self, markets: list[Market]) -> list[ArbitrageOpportunity]:
        """Run all arbitrage checks on a list of markets."""
        all_opps: list[ArbitrageOpportunity] = []

        # 1. Single condition arbitrage
        for market in markets:
            all_opps.extend(self.check_single_condition(market))

        # 2. Market rebalancing (NegRisk markets)
        neg_risk_markets = [m for m in markets if m.is_multi_condition]
        for market in neg_risk_markets:
            all_opps.extend(self.check_market_rebalancing(market))

        # 3. Combinatorial arbitrage between dependent markets
        pairs = self.find_dependent_pairs(markets)
        for m1, m2, deps in pairs:
            all_opps.extend(self.check_combinatorial_arbitrage(m1, m2, deps))

        # Sort by profit descending
        all_opps.sort(key=lambda o: o.profit_per_dollar, reverse=True)
        return all_opps

    # ================================================================== #
    #  Helpers
    # ================================================================== #

    def _is_uncertain(self, condition: Condition) -> bool:
        """Paper Section 6: only analyze when no position > MAX_CONDITION_PRICE."""
        return condition.yes_price <= MAX_CONDITION_PRICE

    @staticmethod
    def _group_by_date(markets: list[Market]) -> dict[str, list[Market]]:
        """Group markets by end date. Skip markets without dates."""
        groups: dict[str, list[Market]] = {}
        for m in markets:
            if m.end_date:
                key = m.end_date[:10]
                groups.setdefault(key, []).append(m)
        return groups

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Simple text similarity using SequenceMatcher."""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    @staticmethod
    def _find_dependent_conditions_heuristic(
        m1: Market, m2: Market
    ) -> list[tuple[Condition, Condition]]:
        """
        Heuristic fallback: match conditions by question text similarity.
        Used when no LLM is configured.
        """
        deps = []
        for c1 in m1.conditions:
            for c2 in m2.conditions:
                sim = SequenceMatcher(
                    None, c1.question.lower(), c2.question.lower()
                ).ratio()
                if sim > 0.6:
                    deps.append((c1, c2))
        return deps
