"""
Configuration for Polymarket Arbitrage Bot.
Based on: "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"
(Saguillo et al., 2025)
"""

# --- API Endpoints ---
CLOB_BASE_URL = "https://clob.polymarket.com"
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

# --- Arbitrage Detection Parameters ---
# Minimum profit margin to consider an opportunity (Definition 3 in paper)
# The paper uses 0.05 ($0.05 on the dollar) for their analysis
MIN_PROFIT_MARGIN = 0.02

# Maximum price for any single condition to be considered "uncertain"
# Paper uses 0.95 - conditions above this are basically resolved
MAX_CONDITION_PRICE = 0.95

# --- Market Filters ---
# Only analyze active, open markets
MARKET_FILTERS = {
    "active": True,
    "closed": False,
}

# Topics from Polymarket (used in paper Section 4.1.1)
TOPICS = ["Politics", "Economy", "Technology", "Crypto", "Twitter", "Culture", "Sports"]

# --- Scanning ---
# How many markets to fetch per page
MARKETS_PER_PAGE = 100

# Delay between API calls (seconds) to respect rate limits
API_DELAY = 0.1

# --- Display ---
# Minimum profit in USD to display an opportunity
MIN_DISPLAY_PROFIT_USD = 1.0
