"""
Topic classification and semantic similarity using sentence embeddings.

Implements paper Section 4.1.1:
  - Classify markets into topics using cosine similarity with topic embeddings
  - Pre-filter market pairs by topic + date before expensive LLM calls

The paper uses Linq-Embed-Mistral. We use sentence-transformers which is
local, free, and fast enough for our scale.
"""
import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from config import TOPICS, EMBEDDING_MODEL, EMBEDDING_SIMILARITY_THRESHOLD
from models import Market

logger = logging.getLogger(__name__)


class TopicClassifier:
    """Classifies markets into topics and computes semantic similarity."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        topics: list[str] = None,
    ):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.topics = topics or TOPICS

        # Pre-compute topic embeddings
        self._topic_embeddings = self.model.encode(
            self.topics, normalize_embeddings=True
        )
        # Cache for market question embeddings
        self._cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    #  Topic Classification (Paper Section 4.1.1)
    # ------------------------------------------------------------------ #

    def classify_market(self, market: Market) -> str:
        """
        Assign a topic to a market based on cosine similarity
        between question embedding and topic embeddings.
        """
        emb = self._get_embedding(market.question)
        similarities = emb @ self._topic_embeddings.T
        best_idx = int(np.argmax(similarities))
        return self.topics[best_idx]

    def classify_markets(self, markets: list[Market]) -> dict[str, str]:
        """Classify all markets. Returns {market_id: topic}."""
        result = {}
        questions = [m.question for m in markets]

        # Batch encode for efficiency
        embeddings = self.model.encode(questions, normalize_embeddings=True)
        for i, market in enumerate(markets):
            self._cache[market.question] = embeddings[i]
            similarities = embeddings[i] @ self._topic_embeddings.T
            best_idx = int(np.argmax(similarities))
            result[market.market_id] = self.topics[best_idx]

        return result

    # ------------------------------------------------------------------ #
    #  Semantic Similarity Between Markets
    # ------------------------------------------------------------------ #

    def compute_similarity(self, m1: Market, m2: Market) -> float:
        """Cosine similarity between two market questions (0 to 1)."""
        e1 = self._get_embedding(m1.question)
        e2 = self._get_embedding(m2.question)
        return float(e1 @ e2)

    # ------------------------------------------------------------------ #
    #  Grouping for Efficient Pairwise Comparison
    # ------------------------------------------------------------------ #

    def group_by_topic_and_date(
        self, markets: list[Market]
    ) -> dict[tuple[str, str], list[Market]]:
        """
        Group markets by (topic, end_date) for efficient filtering.

        Paper Section 4.1.1: "We consider only pairs of markets within
        a given topic and end date to limit our search space."
        """
        # Batch classify all markets
        topics = self.classify_markets(markets)

        groups: dict[tuple[str, str], list[Market]] = {}
        for market in markets:
            if not market.end_date:
                continue
            topic = topics.get(market.market_id, "unknown")
            date_key = market.end_date[:10]
            key = (topic, date_key)
            groups.setdefault(key, []).append(market)

        logger.info(
            f"Grouped {len(markets)} markets into {len(groups)} (topic, date) groups"
        )
        return groups

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, using cache."""
        if text not in self._cache:
            self._cache[text] = self.model.encode(
                text, normalize_embeddings=True
            )
        return self._cache[text]
