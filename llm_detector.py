"""
LLM-based dependency detection between Polymarket markets.

Implements paper Sections 5.1 (single market validation) and 5.2 (pairwise detection).
Uses an OpenAI-compatible API, so works with:
  - Ollama (local): http://localhost:11434/v1
  - DeepSeek API:   https://api.deepseek.com/v1
  - OpenAI:         https://api.openai.com/v1
"""
import json
import logging
import time
from dataclasses import dataclass, field

from openai import OpenAI

from config import (
    LLM_API_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
    MAX_CONDITIONS_PER_MARKET,
)
from models import Market, Condition
from llm_prompts import build_pair_detection_prompt, build_single_market_prompt

logger = logging.getLogger(__name__)


@dataclass
class DependencyResult:
    """Result of LLM dependency check between two markets."""
    market1_id: str
    market2_id: str
    is_dependent: bool
    valid_combinations: list[list[bool]] = field(default_factory=list)
    n_conditions_m1: int = 0
    n_conditions_m2: int = 0
    dependent_subsets: list[tuple[list[int], list[int]]] = field(default_factory=list)
    reasoning: str = ""


class LLMDependencyDetector:
    """Detects logical dependencies between markets using an LLM."""

    def __init__(
        self,
        api_base_url: str = LLM_API_BASE_URL,
        api_key: str = LLM_API_KEY,
        model_name: str = LLM_MODEL,
    ):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=api_base_url,
            api_key=api_key or "not-needed",
            timeout=LLM_TIMEOUT,
        )

    # ------------------------------------------------------------------ #
    #  Single Market Validation (Section 5.1)
    # ------------------------------------------------------------------ #

    def check_single_market(self, market: Market) -> bool:
        """
        Validate that the LLM correctly identifies mutual exclusivity
        within a single market's conditions.

        Returns True if validation passes (exactly n vectors, each with
        exactly one True).
        """
        conditions = self._reduce_conditions(market)
        indexed = [(i, c.question) for i, c in enumerate(conditions)]
        prompt = build_single_market_prompt(indexed)

        response = self._call_llm(prompt)
        if response is None:
            return False

        combos = response.get("valid_combinations", [])
        return self._validate_single_market(combos, len(conditions))

    # ------------------------------------------------------------------ #
    #  Pairwise Market Dependency (Section 5.2)
    # ------------------------------------------------------------------ #

    def check_market_pair(self, m1: Market, m2: Market) -> DependencyResult:
        """
        Check if two markets have dependent conditions using LLM reasoning.

        The LLM returns all valid truth-value combinations for conditions
        from both markets. If |combinations| < n * m, markets are dependent.
        """
        conds_m1 = self._reduce_conditions(m1)
        conds_m2 = self._reduce_conditions(m2)

        n, m = len(conds_m1), len(conds_m2)

        # Build indexed conditions (m1: 0..n-1, m2: n..n+m-1)
        indexed_m1 = [(i, c.question) for i, c in enumerate(conds_m1)]
        indexed_m2 = [(n + j, c.question) for j, c in enumerate(conds_m2)]

        prompt = build_pair_detection_prompt(indexed_m1, indexed_m2)
        response = self._call_llm(prompt)

        result = DependencyResult(
            market1_id=m1.market_id,
            market2_id=m2.market_id,
            is_dependent=False,
            n_conditions_m1=n,
            n_conditions_m2=m,
        )

        if response is None:
            return result

        combos = response.get("valid_combinations", [])

        # Validation checks (paper Section 5.2):
        # (i) Valid JSON - already parsed
        # (ii) Exactly one true in each market's condition set per vector
        # (iii) Number of vectors <= n * m
        if not self._validate_pair(combos, n, m):
            logger.warning(
                f"LLM validation failed for pair {m1.market_id} x {m2.market_id}"
            )
            return result

        result.valid_combinations = combos

        # Independent if |combos| == n * m
        if len(combos) < n * m:
            result.is_dependent = True
            result.dependent_subsets = self._find_equivalent_pairs(combos, n, m)
            logger.debug(
                f"Dependent pair found: {m1.question[:50]} <-> {m2.question[:50]} "
                f"({len(combos)} combos vs {n * m} independent)"
            )

        return result

    # ------------------------------------------------------------------ #
    #  Condition Reduction (Paper Section 5.1, Appendix C)
    # ------------------------------------------------------------------ #

    def _reduce_conditions(self, market: Market) -> list[Condition]:
        """
        Reduce market to at most MAX_CONDITIONS_PER_MARKET conditions.
        Keep top conditions by volume, merge rest into "Other".
        Paper shows >90% of liquidity is in top 4 conditions.
        """
        if len(market.conditions) <= MAX_CONDITIONS_PER_MARKET:
            return market.conditions

        # Sort by YES price as proxy for activity (higher price = more traded)
        sorted_conds = sorted(
            market.conditions, key=lambda c: c.yes_price, reverse=True
        )

        top = sorted_conds[: MAX_CONDITIONS_PER_MARKET]

        # Create "Other" catch-all condition
        other_yes = sum(c.yes_price for c in sorted_conds[MAX_CONDITIONS_PER_MARKET:])
        other_yes = min(other_yes, 1.0)
        other = Condition(
            condition_id="other",
            question="None of the above",
            tokens=[],
        )
        # Manually set a synthetic price
        from models import Token
        other.tokens = [
            Token(token_id="", outcome="Yes", price=other_yes),
            Token(token_id="", outcome="No", price=1.0 - other_yes),
        ]

        return top + [other]

    # ------------------------------------------------------------------ #
    #  Validation (Paper Section 5.1-5.2)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_single_market(combos: list[list[bool]], n: int) -> bool:
        """
        Validate single market LLM output:
        (i) Correct number of vectors (exactly n)
        (ii) Each vector has exactly one True
        """
        if len(combos) != n:
            return False
        for combo in combos:
            if len(combo) != n:
                return False
            if sum(1 for v in combo if v) != 1:
                return False
        return True

    @staticmethod
    def _validate_pair(combos: list[list[bool]], n: int, m: int) -> bool:
        """
        Validate pairwise LLM output (paper Section 5.2):
        (i) Valid structure
        (ii) Exactly one True in positions 0..n-1 (market 1)
        (iii) Exactly one True in positions n..n+m-1 (market 2)
        (iv) Number of vectors <= n * m
        """
        if not combos or len(combos) > n * m:
            return False

        for combo in combos:
            if len(combo) != n + m:
                return False
            # Check market 1: exactly one True in first n positions
            m1_trues = sum(1 for v in combo[:n] if v)
            if m1_trues != 1:
                return False
            # Check market 2: exactly one True in remaining m positions
            m2_trues = sum(1 for v in combo[n:] if v)
            if m2_trues != 1:
                return False

        return True

    @staticmethod
    def _find_equivalent_pairs(
        combos: list[list[bool]], n: int, m: int
    ) -> list[tuple[list[int], list[int]]]:
        """
        From valid combinations, find biconditional equivalent pairs.
        Only returns pairs where i in M1 and j in M2 always co-occur:
        i=True => exactly j=True, AND j=True => exactly i=True.
        """
        # m1->m2: for each i, when i=True, which single j is always True?
        m1_forces_m2: dict[int, int | None] = {}
        for i in range(n):
            m2_when_i: set[int] = set()
            for combo in combos:
                if combo[i]:
                    for j in range(n, n + m):
                        if combo[j]:
                            m2_when_i.add(j - n)
            m1_forces_m2[i] = next(iter(m2_when_i)) if len(m2_when_i) == 1 else None

        # m2->m1: inverse
        m2_forces_m1: dict[int, int | None] = {}
        for j in range(m):
            m1_when_j: set[int] = set()
            for combo in combos:
                if combo[n + j]:
                    for i in range(n):
                        if combo[i]:
                            m1_when_j.add(i)
            m2_forces_m1[j] = next(iter(m1_when_j)) if len(m1_when_j) == 1 else None

        # Only biconditional pairs: i->j AND j->i
        pairs: list[tuple[list[int], list[int]]] = []
        for i in range(n):
            j = m1_forces_m2[i]
            if j is not None and m2_forces_m1.get(j) == i:
                pairs.append(([i], [j]))
        return pairs

    # ------------------------------------------------------------------ #
    #  LLM API Call
    # ------------------------------------------------------------------ #

    def _call_llm(self, prompt: str) -> dict | None:
        """Call LLM API with retry logic. Returns parsed JSON or None."""
        for attempt in range(LLM_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = response.choices[0].message.content.strip()

                # Extract JSON from possible markdown code blocks
                content = self._extract_json(content)

                return json.loads(content)

            except json.JSONDecodeError as e:
                logger.warning(f"LLM returned invalid JSON (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.warning(f"LLM API error (attempt {attempt + 1}): {e}")

            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

        return None

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from markdown code blocks if present."""
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end].strip()
        return text
