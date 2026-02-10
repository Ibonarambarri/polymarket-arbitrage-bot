#!/usr/bin/env python3
"""
Polymarket Arbitrage Scanner

Scans Polymarket prediction markets for arbitrage opportunities.
Based on: "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"
          (Saguillo, Ghafouri, Kiffer, Suarez-Tangil, 2025)

Detects three types of arbitrage:
  1. Single Condition  - YES + NO prices deviate from $1 within one condition
  2. Market Rebalancing - sum of YES prices across a NegRisk market != $1
  3. Combinatorial      - pricing mismatch between dependent markets

Uses a local LLM (Ollama + deepseek-r1) by default for dependency detection.
Falls back to heuristic mode if Ollama is not available.

Usage:
  python main.py                     # LLM mode (Ollama local, default)
  python main.py --no-llm            # Heuristic mode only (no LLM)
  python main.py --llm-model mistral # Use a different model
  python main.py --spark              # + PySpark parallel processing (scale)
  python main.py --refresh            # Live CLOB prices
  python main.py --min-margin 0.05    # Custom profit threshold
  python main.py --live               # Continuous scanning (every 60s)
  python main.py --live --interval 30 # Continuous scanning every 30s
"""
import argparse
import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

# Prevent transformers from loading TensorFlow (not needed, avoids Keras version conflicts)
os.environ.setdefault("USE_TF", "0")
# Suppress tokenizers fork parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from tqdm import tqdm

from config import (
    MIN_PROFIT_MARGIN,
    MIN_DISPLAY_PROFIT_USD,
    LLM_ENABLED,
    LLM_API_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL,
    SPARK_ENABLED,
    SPARK_MIN_MARKETS,
    DEFAULT_SCAN_INTERVAL,
)
from models import ArbitrageOpportunity, ArbitrageType
from fetcher import PolymarketFetcher
from arbitrage import ArbitrageDetector

# Module logger
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Display helpers
# ------------------------------------------------------------------ #

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
    "dim": "\033[2m",
}


def c(text: str, color: str) -> str:
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def print_header(use_llm: bool, use_spark: bool, live: bool = False):
    print()
    print(c("=" * 70, "cyan"))
    print(c("  POLYMARKET ARBITRAGE SCANNER", "bold"))
    print(c("  Based on: 'Unravelling the Probabilistic Forest' (2025)", "dim"))
    modes = []
    if use_llm:
        modes.append("LLM")
    if use_spark:
        modes.append("Spark")
    if live:
        modes.append("LIVE")
    if modes:
        print(c(f"  Mode: {' + '.join(modes)}", "magenta"))
    print(c("=" * 70, "cyan"))
    print()


def print_opportunity(opp: ArbitrageOpportunity, idx: int):
    type_colors = {
        ArbitrageType.SINGLE_CONDITION: "yellow",
        ArbitrageType.MARKET_REBALANCING_LONG: "green",
        ArbitrageType.MARKET_REBALANCING_SHORT: "red",
        ArbitrageType.COMBINATORIAL: "magenta",
    }
    color = type_colors.get(opp.arb_type, "cyan")

    print(c(f"\n{'─' * 60}", "dim"))
    print(c(f"  #{idx + 1} | {opp.arb_type.value.upper()}", color))
    print(c(f"  Market: ", "bold") + opp.market.question[:80])
    if opp.second_market:
        print(c(f"  Market2: ", "bold") + opp.second_market.question[:80])
    print(c(f"  Profit/unit: ", "bold") + c(f"${abs(opp.profit_per_dollar):.4f}", "green"))
    print(c(f"  Strategy: ", "bold") + opp.strategy)
    print(c(f"  Liquidity: ", "bold") + f"${opp.market.liquidity:,.0f}")
    print(c(f"  Est. max profit: ", "bold") + c(f"${opp.estimated_max_profit_usd:,.2f}", "green"))
    print()
    for line in opp.details.split("\n"):
        print(f"    {line}")


def print_summary(opportunities: list[ArbitrageOpportunity], duration: float):
    print(c(f"\n{'=' * 60}", "cyan"))
    print(c("  SCAN SUMMARY", "bold"))
    print(c(f"{'=' * 60}", "cyan"))

    by_type: dict[ArbitrageType, list] = {}
    for opp in opportunities:
        by_type.setdefault(opp.arb_type, []).append(opp)

    total_profit = sum(o.estimated_max_profit_usd for o in opportunities)

    print(f"\n  Total opportunities: {c(str(len(opportunities)), 'bold')}")
    for arb_type, opps in by_type.items():
        print(f"    {arb_type.value}: {len(opps)}")
    print(f"  Total estimated profit potential: {c(f'${total_profit:,.2f}', 'green')}")
    print(f"  Scan duration: {duration:.1f}s")
    print(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(c("  Note: Profits estimated from midpoint prices. Actual execution costs may differ.", "dim"))
    print()


# ------------------------------------------------------------------ #
#  Ollama availability check
# ------------------------------------------------------------------ #

def check_ollama(api_base: str, model: str) -> bool:
    """Check that Ollama is running and the required model is available.

    1. Pings Ollama's /api/tags endpoint.
    2. If the model is not listed, attempts an auto-pull.
    3. Returns True if ready, False otherwise.
    """
    # Derive Ollama host from the OpenAI-compat base URL (strip /v1 suffix)
    host = api_base.rstrip("/")
    if host.endswith("/v1"):
        host = host[:-3]

    # 1. Check Ollama is running
    tags_url = f"{host}/api/tags"
    try:
        req = urllib.request.Request(tags_url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        logger.warning(f"Ollama not reachable at {host}: {e}")
        print(c(f"  Ollama is not running at {host}", "yellow"))
        print(c("  Start it with: ollama serve", "yellow"))
        return False

    # 2. Check if the model is available
    available_models = [m.get("name", "") for m in data.get("models", [])]
    # Match with or without tag suffix (e.g. "deepseek-r1:latest" matches "deepseek-r1")
    model_found = any(
        m == model or m.startswith(model + ":") or model.startswith(m.split(":")[0] + ":")
        for m in available_models
    )

    if model_found:
        logger.info(f"Model '{model}' is available in Ollama")
        return True

    # 3. Auto-pull the model
    print(c(f"  Model '{model}' not found locally. Pulling from Ollama registry...", "cyan"))
    logger.info(f"Auto-pulling model '{model}' from Ollama")
    pull_url = f"{host}/api/pull"
    pull_body = json.dumps({"name": model}).encode()
    try:
        req = urllib.request.Request(
            pull_url, data=pull_body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            last_status = ""
            for line in resp:
                try:
                    chunk = json.loads(line.decode().strip())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                status = chunk.get("status", "")
                if status != last_status:
                    print(c(f"    {status}", "dim"))
                    last_status = status
                if chunk.get("error"):
                    print(c(f"  Pull failed: {chunk['error']}", "yellow"))
                    return False
        print(c(f"  Model '{model}' pulled successfully.", "green"))
        logger.info(f"Model '{model}' pulled successfully")
        return True
    except (urllib.error.URLError, OSError) as e:
        logger.warning(f"Failed to pull model '{model}': {e}")
        print(c(f"  Failed to pull model '{model}': {e}", "yellow"))
        return False


# ------------------------------------------------------------------ #
#  Opportunity tracker (for live mode)
# ------------------------------------------------------------------ #

class OpportunityTracker:
    """Tracks arbitrage opportunities across scan cycles to report only new ones."""

    def __init__(self):
        self.seen_keys: set[str] = set()
        self.last_cycle_keys: set[str] = set()
        self.total_ever_found: int = 0

    @staticmethod
    def _make_key(opp: ArbitrageOpportunity) -> str:
        """Generate a stable key for an opportunity based on type, market, and conditions."""
        parts = [opp.arb_type.value, opp.market.market_id]
        parts.extend(sorted(cond.condition_id for cond in opp.conditions_involved))
        if opp.second_market:
            parts.append(opp.second_market.market_id)
        return "|".join(parts)

    def process_cycle(
        self, opportunities: list[ArbitrageOpportunity]
    ) -> tuple[list[ArbitrageOpportunity], int]:
        """Process a scan cycle's results.

        Returns (new_opportunities, disappeared_count).
        """
        current_keys: set[str] = set()
        new_opps: list[ArbitrageOpportunity] = []

        for opp in opportunities:
            key = self._make_key(opp)
            current_keys.add(key)
            if key not in self.seen_keys:
                new_opps.append(opp)

        disappeared = len(self.last_cycle_keys - current_keys)
        self.seen_keys.update(current_keys)
        self.last_cycle_keys = current_keys
        self.total_ever_found += len(new_opps)

        return new_opps, disappeared


# ------------------------------------------------------------------ #
#  Scanner components
# ------------------------------------------------------------------ #

def _initialize_components(
    use_llm: bool,
    use_spark: bool,
    llm_api_base: str | None,
    llm_model: str | None,
    min_margin: float,
) -> dict:
    """Create and initialize all scanner components (fetcher, detector, optional LLM/Spark).

    Returns a dict with keys: fetcher, detector, spark, use_llm (possibly updated).
    """
    logger.info("Initializing PolymarketFetcher")
    fetcher = PolymarketFetcher()

    llm_detector = None
    topic_classifier = None
    effective_use_llm = use_llm

    if use_llm:
        print(c("  [0/4] Initializing LLM + embeddings...", "cyan"))
        logger.info("Initializing LLM-based dependency detection")

        api_base = llm_api_base or LLM_API_BASE_URL
        model = llm_model or LLM_MODEL

        if not check_ollama(api_base, model):
            print(c("         Falling back to heuristic mode.", "yellow"))
            effective_use_llm = False
        else:
            try:
                from llm_detector import LLMDependencyDetector
                from embeddings import TopicClassifier

                logger.info(f"Connecting to LLM API: {api_base}, model: {model}")
                llm_detector = LLMDependencyDetector(
                    api_base_url=api_base,
                    api_key=LLM_API_KEY,
                    model_name=model,
                )
                logger.info("Loading topic classifier embeddings")
                topic_classifier = TopicClassifier()
                print(f"         LLM: {model} @ {api_base}")
                print(f"         Embeddings: loaded")
                logger.info("LLM initialization completed successfully")
            except (ImportError, ValueError, OSError) as e:
                logger.warning(f"LLM dependencies not available: {e}")
                print(c(f"         Warning: {e}", "yellow"))
                print(c("         Install deps: pip install openai sentence-transformers", "yellow"))
                print(c("         Falling back to heuristic mode.", "yellow"))
                effective_use_llm = False

    detector = ArbitrageDetector(
        min_margin=min_margin,
        llm_detector=llm_detector,
        topic_classifier=topic_classifier,
    )

    spark = None
    if use_spark:
        try:
            from spark_analyzer import SparkArbitrageAnalyzer
            spark = SparkArbitrageAnalyzer()
            logger.info("PySpark initialized")
        except ImportError as e:
            logger.warning(f"PySpark not available: {e}")
            print(c("         Warning: pyspark not installed. Using sequential mode.", "yellow"))

    return {
        "fetcher": fetcher,
        "detector": detector,
        "spark": spark,
        "use_llm": effective_use_llm,
    }


def _run_single_scan(
    components: dict,
    refresh_prices: bool,
    min_margin: float,
) -> tuple[list[ArbitrageOpportunity], float]:
    """Execute a single scan cycle: fetch, parse, detect, filter.

    Returns (opportunities, duration_seconds).
    """
    fetcher = components["fetcher"]
    detector = components["detector"]
    spark = components["spark"]

    # 1. Fetch all active events
    print(c("  [1/4] Fetching active events from Polymarket...", "cyan"))
    logger.info("Fetching active events from Polymarket Gamma API")
    events = fetcher.get_all_active_events()
    logger.info(f"Successfully fetched {len(events)} active events")
    print(f"         Found {len(events)} active events")

    # 2. Parse into Market objects
    print(c("  [2/4] Parsing markets and conditions...", "cyan"))
    logger.info(f"Parsing {len(events)} events into Market objects")
    all_markets = []
    for event in tqdm(events, desc="         Parsing events", unit="event", leave=False, disable=None):
        markets = fetcher.parse_event_to_markets(event)
        all_markets.extend(markets)

    neg_risk_count = sum(1 for m in all_markets if m.is_multi_condition)
    single_count = len(all_markets) - neg_risk_count
    total_conditions = sum(len(m.conditions) for m in all_markets)
    logger.info(f"Parsed {len(all_markets)} markets: {neg_risk_count} NegRisk, {single_count} single, {total_conditions} total conditions")
    print(f"         {len(all_markets)} markets ({neg_risk_count} NegRisk, {single_count} single)")
    print(f"         {total_conditions} total conditions")

    # 3. Optionally refresh prices
    if refresh_prices:
        print(c("  [3/4] Refreshing live prices from CLOB API...", "cyan"))
        logger.info(f"Refreshing live prices for {len(all_markets)} markets from CLOB API")
        for market in tqdm(all_markets, desc="         Refreshing prices", unit="market", leave=False, disable=None):
            fetcher.refresh_prices(market)
        logger.info("Price refresh completed")
    else:
        print(c("  [3/4] Using Gamma API prices (use --refresh for live CLOB prices)", "dim"))
        logger.info("Using Gamma API prices (no refresh)")

    # 4. Detect arbitrage
    print(c("  [4/4] Scanning for arbitrage opportunities...", "cyan"))
    logger.info("Starting arbitrage detection")
    start = time.time()

    if spark and len(all_markets) >= SPARK_MIN_MARKETS:
        print(c("         Using PySpark parallel processing...", "magenta"))
        logger.info(f"Using PySpark parallel processing for {len(all_markets)} markets")
        opportunities = spark.analyze_parallel(all_markets, detector)
        logger.info("PySpark analysis completed")
    else:
        if spark and len(all_markets) < SPARK_MIN_MARKETS:
            print(c(f"         Skipping Spark (<{SPARK_MIN_MARKETS} markets). Using sequential.", "dim"))
            logger.info(f"Market count ({len(all_markets)}) below Spark threshold ({SPARK_MIN_MARKETS}), using sequential mode")
        logger.info(f"Scanning {len(all_markets)} markets sequentially")
        opportunities = detector.scan_all(all_markets)

    duration = time.time() - start
    logger.info(f"Arbitrage detection completed in {duration:.2f}s, found {len(opportunities)} raw opportunities")

    # Filter by minimum display threshold
    opportunities_before_filter = len(opportunities)
    opportunities = [
        o for o in opportunities
        if o.estimated_max_profit_usd >= MIN_DISPLAY_PROFIT_USD
    ]
    filtered_count = opportunities_before_filter - len(opportunities)
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} opportunities below ${MIN_DISPLAY_PROFIT_USD} threshold")

    return opportunities, duration


# ------------------------------------------------------------------ #
#  Main scanner (single scan)
# ------------------------------------------------------------------ #

def scan(
    refresh_prices: bool = False,
    min_margin: float = MIN_PROFIT_MARGIN,
    use_llm: bool = False,
    use_spark: bool = False,
    llm_api_base: str | None = None,
    llm_model: str | None = None,
):
    logger.info("Starting Polymarket arbitrage scan")
    logger.info(f"Config: refresh_prices={refresh_prices}, min_margin={min_margin}, use_llm={use_llm}, use_spark={use_spark}")

    print_header(use_llm, use_spark)

    components = _initialize_components(use_llm, use_spark, llm_api_base, llm_model, min_margin)

    try:
        opportunities, duration = _run_single_scan(components, refresh_prices, min_margin)

        # Display results
        if not opportunities:
            print(c("\n  No arbitrage opportunities found above threshold.", "yellow"))
            print(f"  (min margin: ${min_margin}, min display: ${MIN_DISPLAY_PROFIT_USD})")
            logger.info("No arbitrage opportunities found above display threshold")
        else:
            print(c(f"\n  Found {len(opportunities)} opportunities!", "green"))
            logger.info(f"Found {len(opportunities)} arbitrage opportunities above threshold")
            for idx, opp in enumerate(opportunities):
                print_opportunity(opp, idx)
                logger.debug(f"Opportunity #{idx+1}: {opp.arb_type.value} in market {opp.market.market_id}")

        print_summary(opportunities, duration)
        logger.info("Scan completed successfully")
        return opportunities
    finally:
        if components["spark"]:
            components["spark"].stop()


# ------------------------------------------------------------------ #
#  Live scanner (continuous mode)
# ------------------------------------------------------------------ #

def scan_live(
    refresh_prices: bool = False,
    min_margin: float = MIN_PROFIT_MARGIN,
    use_llm: bool = False,
    use_spark: bool = False,
    llm_api_base: str | None = None,
    llm_model: str | None = None,
    interval: int = DEFAULT_SCAN_INTERVAL,
):
    logger.info(f"Starting live scan mode (interval={interval}s)")

    print_header(use_llm, use_spark, live=True)
    print(c(f"  Scanning every {interval}s. Press Ctrl+C to stop.", "dim"))
    print()

    components = _initialize_components(use_llm, use_spark, llm_api_base, llm_model, min_margin)
    tracker = OpportunityTracker()
    cycle = 0

    try:
        while True:
            cycle += 1
            cycle_start = time.time()
            timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(c(f"\n{'=' * 60}", "cyan"))
            print(c(f"  CYCLE #{cycle} | {timestamp} UTC", "bold"))
            print(c(f"{'=' * 60}", "cyan"))

            try:
                opportunities, duration = _run_single_scan(components, refresh_prices, min_margin)
            except Exception as exc:
                logger.error(f"Cycle #{cycle} failed: {exc}", exc_info=True)
                print(c(f"\n  Cycle #{cycle} error: {exc}", "red"))
                print(c("  Will retry next cycle.", "yellow"))
                elapsed = time.time() - cycle_start
                time.sleep(max(0, interval - elapsed))
                continue

            new_opps, disappeared = tracker.process_cycle(opportunities)

            if new_opps:
                print(c(f"\n  {len(new_opps)} NEW opportunity(ies)!", "green"))
                for idx, opp in enumerate(new_opps):
                    print_opportunity(opp, idx)
            else:
                print(c("\n  No new opportunities this cycle.", "dim"))

            # Cycle summary
            print(c(f"\n  {'─' * 40}", "dim"))
            print(f"  Cycle: {c(str(len(opportunities)), 'bold')} total | "
                  f"{c(str(len(new_opps)), 'green')} new | "
                  f"{c(str(disappeared), 'yellow')} disappeared")
            print(f"  All-time unique: {c(str(tracker.total_ever_found), 'bold')} | "
                  f"Scan: {duration:.1f}s")

            elapsed = time.time() - cycle_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                print(c(f"  Next scan in {sleep_time:.0f}s...", "dim"))
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(c(f"\n\n{'=' * 60}", "cyan"))
        print(c("  LIVE MODE STOPPED", "bold"))
        print(c(f"{'=' * 60}", "cyan"))
        print(f"  Cycles completed: {cycle}")
        print(f"  Total unique opportunities found: {c(str(tracker.total_ever_found), 'green')}")
        print(f"  Last cycle had: {len(tracker.last_cycle_keys)} active opportunities")
        print()
    finally:
        if components["spark"]:
            components["spark"].stop()


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Arbitrage Scanner - detect mispriced prediction markets"
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Refresh prices from CLOB API (slower but more accurate)",
    )
    parser.add_argument(
        "--min-margin", type=float, default=MIN_PROFIT_MARGIN,
        help=f"Minimum profit margin to report (default: {MIN_PROFIT_MARGIN})",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Force enable LLM dependency detection (enabled by default via config)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable LLM dependency detection, use heuristic mode only",
    )
    parser.add_argument(
        "--spark", action="store_true",
        help="Use PySpark for parallel processing",
    )
    parser.add_argument(
        "--llm-api-base", type=str, default=None,
        help="LLM API base URL (overrides config/env)",
    )
    parser.add_argument(
        "--llm-model", type=str, default=None,
        help="LLM model name (overrides config/env)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Continuous scanning mode (re-scans every --interval seconds)",
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_SCAN_INTERVAL,
        help=f"Seconds between scans in --live mode (default: {DEFAULT_SCAN_INTERVAL})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress INFO logs (only show warnings/errors)",
    )
    args = parser.parse_args()

    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    use_llm = not args.no_llm and (args.llm or LLM_ENABLED)
    use_spark = args.spark or SPARK_ENABLED

    try:
        if args.live:
            scan_live(
                refresh_prices=args.refresh,
                min_margin=args.min_margin,
                use_llm=use_llm,
                use_spark=use_spark,
                llm_api_base=args.llm_api_base,
                llm_model=args.llm_model,
                interval=args.interval,
            )
        else:
            scan(
                refresh_prices=args.refresh,
                min_margin=args.min_margin,
                use_llm=use_llm,
                use_spark=use_spark,
                llm_api_base=args.llm_api_base,
                llm_model=args.llm_model,
            )
    except KeyboardInterrupt:
        print(c("\n\n  Scan interrupted by user.", "yellow"))
        sys.exit(0)


if __name__ == "__main__":
    main()
