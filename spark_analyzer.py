"""
PySpark-based parallel arbitrage analysis.

Enables fast parallel processing of large market datasets.
Useful when scanning thousands of markets simultaneously.

Usage:
  python main.py --spark              # Spark without LLM
  python main.py --llm --spark        # Spark + LLM (full power)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    FloatType,
    BooleanType,
    ArrayType,
)

from config import SPARK_MASTER, MIN_PROFIT_MARGIN, MAX_CONDITION_PRICE
from models import Market, Condition, Token, ArbitrageOpportunity, ArbitrageType

if TYPE_CHECKING:
    from arbitrage import ArbitrageDetector

logger = logging.getLogger(__name__)


# Schema for market conditions in Spark DataFrame
CONDITION_SCHEMA = StructType([
    StructField("market_id", StringType(), False),
    StructField("market_question", StringType(), True),
    StructField("market_slug", StringType(), True),
    StructField("neg_risk", BooleanType(), True),
    StructField("end_date", StringType(), True),
    StructField("liquidity", FloatType(), True),
    StructField("condition_id", StringType(), True),
    StructField("condition_question", StringType(), True),
    StructField("yes_price", FloatType(), True),
    StructField("no_price", FloatType(), True),
    StructField("yes_token_id", StringType(), True),
    StructField("no_token_id", StringType(), True),
])


class SparkArbitrageAnalyzer:
    """Parallel arbitrage detection using Apache Spark."""

    def __init__(self, master: str = SPARK_MASTER, app_name: str = "PolymarketArbitrage"):
        logger.info(f"Initializing Spark session ({master})...")
        self.spark = (
            SparkSession.builder
            .master(master)
            .appName(app_name)
            .config("spark.driver.memory", "2g")
            .config("spark.sql.adaptive.enabled", "true")
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info("Spark session ready")

    def analyze_parallel(
        self,
        markets: list[Market],
        detector: ArbitrageDetector,
    ) -> list[ArbitrageOpportunity]:
        """
        Run all arbitrage detection in parallel using Spark.

        Pipeline:
        1. Convert markets to DataFrame
        2. Parallel: Single condition checks
        3. Parallel: Market rebalancing checks
        4. Collect results
        """
        all_opportunities: list[ArbitrageOpportunity] = []

        # Convert to Spark DataFrame
        rows = self._markets_to_rows(markets)
        if not rows:
            return []

        df = self.spark.createDataFrame(rows, schema=CONDITION_SCHEMA)
        df.cache()

        n_markets = df.select("market_id").distinct().count()
        n_conditions = df.count()
        logger.info(f"Spark: loaded {n_markets} markets, {n_conditions} conditions")

        # 1. Single condition arbitrage (parallel per condition)
        single_opps = self._check_single_conditions_spark(df, detector.min_margin)
        all_opportunities.extend(single_opps)

        # 2. Market rebalancing (parallel per market)
        rebalancing_opps = self._check_rebalancing_spark(df, detector.min_margin)
        all_opportunities.extend(rebalancing_opps)

        # 3. Combinatorial arbitrage - delegate to detector (LLM calls not parallelizable in Spark)
        pairs = detector.find_dependent_pairs(markets)
        for m1, m2, deps in pairs:
            all_opportunities.extend(
                detector.check_combinatorial_arbitrage(m1, m2, deps)
            )

        df.unpersist()

        all_opportunities.sort(key=lambda o: o.profit_per_dollar, reverse=True)
        logger.info(f"Spark: found {len(all_opportunities)} total opportunities")
        return all_opportunities

    # ------------------------------------------------------------------ #
    #  1. Single Condition Checks (Parallel)
    # ------------------------------------------------------------------ #

    def _check_single_conditions_spark(
        self, df, min_margin: float
    ) -> list[ArbitrageOpportunity]:
        """Check YES + NO != $1 for each condition in parallel."""
        result_df = (
            df
            .withColumn("spread", F.col("yes_price") + F.col("no_price"))
            .withColumn("deviation", F.abs(F.col("spread") - 1.0))
            .filter(F.col("deviation") >= min_margin)
            .filter(F.col("yes_price") <= MAX_CONDITION_PRICE)
        )

        opportunities = []
        for row in result_df.collect():
            spread = row.spread
            profit = abs(spread - 1.0)

            if spread < 1.0:
                detail = (
                    f"YES={row.yes_price:.4f} + NO={row.no_price:.4f} "
                    f"= {spread:.4f} < $1.00 | Buy both -> profit ${profit:.4f}/unit"
                )
            else:
                detail = (
                    f"YES={row.yes_price:.4f} + NO={row.no_price:.4f} "
                    f"= {spread:.4f} > $1.00 | Split & sell -> profit ${profit:.4f}/unit"
                )

            market = Market(
                market_id=row.market_id,
                slug=row.market_slug or "",
                question=row.market_question or "",
                description="",
                neg_risk=row.neg_risk or False,
                liquidity=row.liquidity or 0.0,
            )
            condition = Condition(
                condition_id=row.condition_id or "",
                question=row.condition_question or "",
                tokens=[
                    Token(row.yes_token_id or "", "Yes", row.yes_price),
                    Token(row.no_token_id or "", "No", row.no_price),
                ],
            )

            opportunities.append(ArbitrageOpportunity(
                arb_type=ArbitrageType.SINGLE_CONDITION,
                market=market,
                profit_per_dollar=profit,
                details=detail,
                conditions_involved=[condition],
            ))

        logger.info(f"Spark single-condition: {len(opportunities)} opportunities")
        return opportunities

    # ------------------------------------------------------------------ #
    #  2. Market Rebalancing Checks (Parallel)
    # ------------------------------------------------------------------ #

    def _check_rebalancing_spark(
        self, df, min_margin: float
    ) -> list[ArbitrageOpportunity]:
        """Check sum(YES) != $1 across conditions in each NegRisk market."""
        # Only NegRisk markets with uncertain conditions
        neg_risk_df = (
            df
            .filter(F.col("neg_risk") == True)
            .filter(F.col("yes_price") <= MAX_CONDITION_PRICE)
        )

        # Aggregate per market
        market_agg = (
            neg_risk_df
            .groupBy("market_id", "market_question", "market_slug", "liquidity")
            .agg(
                F.sum("yes_price").alias("yes_sum"),
                F.count("*").alias("n_conditions"),
            )
            .filter(F.col("n_conditions") >= 2)
            .withColumn("deviation", F.abs(F.col("yes_sum") - 1.0))
            .filter(F.col("deviation") >= min_margin)
        )

        opportunities = []
        for row in market_agg.collect():
            yes_sum = row.yes_sum
            profit = abs(yes_sum - 1.0)

            if yes_sum < 1.0:
                arb_type = ArbitrageType.MARKET_REBALANCING_LONG
                detail = (
                    f"LONG: sum(YES) = {yes_sum:.4f} < $1.00\n"
                    f"  Buy 1 YES of each condition -> guaranteed profit ${profit:.4f}/unit\n"
                    f"  Conditions: {row.n_conditions}"
                )
            else:
                arb_type = ArbitrageType.MARKET_REBALANCING_SHORT
                detail = (
                    f"SHORT: sum(YES) = {yes_sum:.4f} > $1.00\n"
                    f"  Buy 1 NO of each condition -> guaranteed profit ${profit:.4f}/unit\n"
                    f"  Conditions: {row.n_conditions}"
                )

            market = Market(
                market_id=row.market_id,
                slug=row.market_slug or "",
                question=row.market_question or "",
                description="",
                neg_risk=True,
                liquidity=row.liquidity or 0.0,
            )

            opportunities.append(ArbitrageOpportunity(
                arb_type=arb_type,
                market=market,
                profit_per_dollar=profit,
                details=detail,
            ))

        logger.info(f"Spark rebalancing: {len(opportunities)} opportunities")
        return opportunities

    # ------------------------------------------------------------------ #
    #  Data Conversion
    # ------------------------------------------------------------------ #

    @staticmethod
    def _markets_to_rows(markets: list[Market]) -> list[Row]:
        """Convert Market objects to Spark Rows."""
        rows = []
        for market in markets:
            for condition in market.conditions:
                yes_token_id = ""
                no_token_id = ""
                for t in condition.tokens:
                    if t.outcome.lower() == "yes":
                        yes_token_id = t.token_id
                    elif t.outcome.lower() == "no":
                        no_token_id = t.token_id

                rows.append(Row(
                    market_id=market.market_id,
                    market_question=market.question,
                    market_slug=market.slug,
                    neg_risk=market.neg_risk,
                    end_date=market.end_date,
                    liquidity=float(market.liquidity),
                    condition_id=condition.condition_id,
                    condition_question=condition.question,
                    yes_price=float(condition.yes_price),
                    no_price=float(condition.no_price),
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                ))
        return rows

    def stop(self):
        """Stop the Spark session."""
        self.spark.stop()
