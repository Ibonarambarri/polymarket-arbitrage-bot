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

Usage:
  python main.py                  # Scan all active markets
  python main.py --refresh        # Also refresh live prices from CLOB
  python main.py --min-margin 0.03  # Set custom minimum profit margin
"""
import argparse
import logging
import sys
import time
from datetime import datetime, timezone

from config import MIN_PROFIT_MARGIN, MIN_DISPLAY_PROFIT_USD
from models import ArbitrageOpportunity, ArbitrageType
from fetcher import PolymarketFetcher
from arbitrage import ArbitrageDetector

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


def print_header():
    print()
    print(c("=" * 70, "cyan"))
    print(c("  POLYMARKET ARBITRAGE SCANNER", "bold"))
    print(c("  Based on: 'Unravelling the Probabilistic Forest' (2025)", "dim"))
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
    print(c(f"  Profit/unit: ", "bold") + c(f"${opp.profit_per_dollar:.4f}", "green"))
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
    print()


# ------------------------------------------------------------------ #
#  Main scanner
# ------------------------------------------------------------------ #

def scan(refresh_prices: bool = False, min_margin: float = MIN_PROFIT_MARGIN):
    print_header()

    fetcher = PolymarketFetcher()
    detector = ArbitrageDetector(min_margin=min_margin)

    # 1. Fetch all active events
    print(c("  [1/4] Fetching active events from Polymarket...", "cyan"))
    events = fetcher.get_all_active_events()
    print(f"         Found {len(events)} active events")

    # 2. Parse into Market objects
    print(c("  [2/4] Parsing markets and conditions...", "cyan"))
    all_markets = []
    for event in events:
        markets = fetcher.parse_event_to_markets(event)
        all_markets.extend(markets)

    neg_risk_count = sum(1 for m in all_markets if m.is_multi_condition)
    single_count = len(all_markets) - neg_risk_count
    total_conditions = sum(len(m.conditions) for m in all_markets)
    print(f"         {len(all_markets)} markets ({neg_risk_count} NegRisk, {single_count} single)")
    print(f"         {total_conditions} total conditions")

    # 3. Optionally refresh prices
    if refresh_prices:
        print(c("  [3/4] Refreshing live prices from CLOB API...", "cyan"))
        for i, market in enumerate(all_markets):
            fetcher.refresh_prices(market)
            if (i + 1) % 10 == 0:
                print(f"         Refreshed {i + 1}/{len(all_markets)} markets...")
    else:
        print(c("  [3/4] Using Gamma API prices (use --refresh for live CLOB prices)", "dim"))

    # 4. Detect arbitrage
    print(c("  [4/4] Scanning for arbitrage opportunities...", "cyan"))
    start = time.time()
    opportunities = detector.scan_all(all_markets)
    duration = time.time() - start

    # Filter by minimum display threshold
    opportunities = [
        o for o in opportunities
        if o.estimated_max_profit_usd >= MIN_DISPLAY_PROFIT_USD
    ]

    # Display results
    if not opportunities:
        print(c("\n  No arbitrage opportunities found above threshold.", "yellow"))
        print(f"  (min margin: ${min_margin}, min display: ${MIN_DISPLAY_PROFIT_USD})")
    else:
        print(c(f"\n  Found {len(opportunities)} opportunities!", "green"))
        for idx, opp in enumerate(opportunities):
            print_opportunity(opp, idx)

    print_summary(opportunities, duration)
    return opportunities


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
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        scan(refresh_prices=args.refresh, min_margin=args.min_margin)
    except KeyboardInterrupt:
        print(c("\n\n  Scan interrupted by user.", "yellow"))
        sys.exit(0)


if __name__ == "__main__":
    main()
