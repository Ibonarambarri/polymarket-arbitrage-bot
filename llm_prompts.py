"""
LLM prompt templates for dependency detection.

Based on Appendix B of the paper:
"Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"
(Saguillo et al., 2025)

The core idea: given a set of binary (True/False) questions from two markets,
ask the LLM to determine all valid logical combinations of truth values.
If the number of valid combinations < n*m (where n,m are condition counts),
the markets are dependent.
"""


def build_pair_detection_prompt(
    conditions_m1: list[tuple[int, str]],
    conditions_m2: list[tuple[int, str]],
) -> str:
    """
    Build the prompt for pairwise market dependency detection (paper Section 5.2).

    Args:
        conditions_m1: [(index, question_text), ...] for market 1
        conditions_m2: [(index, question_text), ...] for market 2

    Returns:
        Formatted prompt string.
    """
    all_conditions = conditions_m1 + conditions_m2
    n = len(all_conditions)

    questions_block = ""
    for idx, question in all_conditions:
        questions_block += f"- ({idx}) {question}\n"

    return f"""You are given a set of binary (True/False) questions. Your task is to determine all valid logical combinations of truth values these questions can take.

Rules:
- Each tuple represents a possible valid assignment of truth values.
- Each tuple must contain exactly {n} values, corresponding to the listed questions.
- The output must be a JSON array where each entry is a list of Boolean values.
- The output must be valid JSON and contain no additional text.

Questions:
{questions_block}
**Expected Output Format:**
```json
{{
  "valid_combinations": [
    [true, false, ...],
    [false, true, ...],
    ...]
}}
```
Ensure the output is strictly formatted as JSON without any additional explanation or formatting artifacts."""


def build_single_market_prompt(conditions: list[tuple[int, str]]) -> str:
    """
    Build the prompt for single market validation (paper Section 5.1).

    Used to verify that the LLM correctly identifies that exactly one
    condition must be True in a market (mutual exclusivity).

    Args:
        conditions: [(index, question_text), ...] for the market

    Returns:
        Formatted prompt string.
    """
    n = len(conditions)

    questions_block = ""
    for idx, question in conditions:
        questions_block += f"- ({idx}) {question}\n"

    return f"""You are given a set of binary (True/False) questions about mutually exclusive outcomes of a single event. Your task is to determine all valid logical combinations of truth values.

Rules:
- Exactly one question must be True in each valid combination (the outcomes are mutually exclusive and exhaustive).
- Each tuple must contain exactly {n} values, corresponding to the listed questions.
- The output must be a JSON array where each entry is a list of Boolean values.
- The output must be valid JSON and contain no additional text.

Questions:
{questions_block}
**Expected Output Format:**
```json
{{
  "valid_combinations": [
    [true, false, false, ...],
    [false, true, false, ...],
    ...]
}}
```
Ensure the output is strictly formatted as JSON without any additional explanation or formatting artifacts."""


CONDITION_REDUCTION_PROMPT = """Given a market with {num_conditions} conditions, the top {keep_top} conditions by trading volume should be kept, with all remaining conditions grouped into a single "Other" category.

Market: "{market_title}"
Conditions (sorted by volume):
{conditions_list}

The "Other" condition represents: if none of the top {keep_top} conditions are True, then "Other" is True.
"""
