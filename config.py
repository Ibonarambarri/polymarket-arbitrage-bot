"""
Configuration for Polymarket Arbitrage Bot.
Based on: "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"
(Saguillo et al., 2025)
"""
import os

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
MARKET_FILTERS = {
    "active": True,
    "closed": False,
}

# Topics from Polymarket (used in paper Section 4.1.1 for pre-filtering)
TOPICS = ["Politics", "Economy", "Technology", "Crypto", "Twitter", "Culture", "Sports"]

# --- Scanning ---
MARKETS_PER_PAGE = 100
API_DELAY = 0.1

# --- Position Sizing ---
MAX_POSITION_SIZE = 10000.0

# --- Display ---
MIN_DISPLAY_PROFIT_USD = 1.0

# --- LLM Configuration (Paper Sections 5.1, 5.2) ---
# Set LLM_ENABLED=True and configure to use LLM-based dependency detection.
# Works with any OpenAI-compatible API (Ollama, DeepSeek, OpenAI, etc.)
LLM_ENABLED = False
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "http://localhost:11434/v1")  # Ollama default
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:latest")
LLM_TIMEOUT = 60
LLM_MAX_RETRIES = 3

# Condition reduction: markets with >N conditions are reduced to top-N + "Other"
# Paper Section 5.1, Appendix C: 90%+ liquidity is in top 4
MAX_CONDITIONS_PER_MARKET = 4

# --- Embedding Configuration (Paper Section 4.1.1) ---
# Used for topic classification and semantic pre-filtering
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_SIMILARITY_THRESHOLD = 0.7  # Min similarity to check pair with LLM

# --- PySpark Configuration ---
# Enable for parallel processing of large market datasets
SPARK_ENABLED = False
SPARK_MASTER = "local[*]"  # Or "spark://host:port" for cluster
SPARK_MIN_MARKETS = 100  # Only use Spark above this threshold
