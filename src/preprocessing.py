"""
preprocessing.py
----------------
Loads and merges the Fear & Greed Index with Hyperliquid historical trade data,
then engineers all features needed downstream for EDA, statistics, and ML.
"""

import pandas as pd
import numpy as np


# ── Sentiment category ordering (used for consistent plot ordering) ──────────
SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

# Numeric encoding maps each category to an integer so ML models can use it
SENTIMENT_ENCODING = {
    "Extreme Fear": 0,
    "Fear": 1,
    "Neutral": 2,
    "Greed": 3,
    "Extreme Greed": 4,
}


def load_raw_data(fear_greed_path: str, trades_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read both CSVs from disk and return them as separate DataFrames.
    No cleaning is done here — that happens in `build_merged_df`.
    """
    sentiment = pd.read_csv(fear_greed_path)
    trades = pd.read_csv(trades_path)
    return sentiment, trades


def clean_sentiment(sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the Fear & Greed Index dataframe:
      - Parse the 'date' column to datetime (it arrives as a string like '2018-02-01').
      - Rename 'classification' → 'Classification' for consistency with the prompt.
      - Keep only columns we actually use.
    """
    df = sentiment.copy()

    # The CSV has a 'date' column (string) and a Unix 'timestamp' column.
    # We use the human-readable date string because it is easier to align with trades.
    df["Date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"classification": "Classification", "value": "SentimentValue"})

    # Drop columns that are redundant after we have Date
    df = df[["Date", "SentimentValue", "Classification"]].drop_duplicates(subset="Date")
    return df


def clean_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the Hyperliquid historical trade dataframe:
      - Parse the 'Timestamp IST' column to datetime.
      - Derive a date-only column for the merge key.
      - Rename columns to remove spaces (cleaner attribute access).
      - Cast numeric columns to float where necessary.
    """
    df = trades.copy()

    # 'Timestamp IST' looks like '02-12-2024 22:50' → parse with dayfirst=True
    df["time"] = pd.to_datetime(df["Timestamp IST"], dayfirst=True, errors="coerce")
    df["Date"] = df["time"].dt.normalize()          # midnight timestamp for merge

    # Friendly column names (no spaces)
    df = df.rename(columns={
        "Account":          "account",
        "Coin":             "coin",
        "Execution Price":  "execution_price",
        "Size Tokens":      "size_tokens",
        "Size USD":         "size_usd",
        "Side":             "side",
        "Direction":        "direction",
        "Closed PnL":       "closedPnL",
        "Fee":              "fee",
        "Crossed":          "crossed",
    })

    # Ensure numeric types
    for col in ["execution_price", "size_tokens", "size_usd", "closedPnL", "fee"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where we cannot determine a date (malformed timestamps)
    df = df.dropna(subset=["Date"])
    return df


def merge_datasets(sentiment: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join trades onto sentiment on 'Date'.
    Trades with no matching sentiment day are dropped (inner-ish behaviour
    via dropna after merge — keeps the merge clean for analysis).
    """
    df = trades.merge(sentiment, on="Date", how="left")
    # Drop rows where sentiment could not be matched
    df = df.dropna(subset=["Classification"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all derived columns needed by EDA, statistics, and ML:

    win                 : 1 if the trade closed in profit, 0 otherwise
    loss                : 1 if the trade closed at a loss
    abs_pnl             : absolute value of closed PnL
    position_value      : notional trade value = execution_price × |size_tokens|
    sentiment_encoded   : ordinal integer encoding of Classification
    sentiment_binary    : simplified Fear (0) vs Greed (1), Neutral is dropped
                          for binary tests but kept in the full dataframe
    log_position_value  : log1p transform to reduce skew in position_value
    """
    df = df.copy()

    # ── Outcome flags ─────────────────────────────────────────────────────────
    df["win"]  = (df["closedPnL"] > 0).astype(int)
    df["loss"] = (df["closedPnL"] < 0).astype(int)
    df["abs_pnl"] = df["closedPnL"].abs()

    # ── Position size ─────────────────────────────────────────────────────────
    # Use Size USD directly — it is already execution_price × |size_tokens|
    # but we keep both for flexibility
    df["position_value"] = df["size_usd"].abs()

    # Log transform to handle extreme outliers in position sizing
    df["log_position_value"] = np.log1p(df["position_value"])

    # ── Sentiment encoding ────────────────────────────────────────────────────
    df["sentiment_encoded"] = df["Classification"].map(SENTIMENT_ENCODING)

    # Binary sentiment: Fear-family (0) vs Greed-family (1); Neutral stays NaN
    fear_labels  = {"Extreme Fear", "Fear"}
    greed_labels = {"Greed", "Extreme Greed"}
    df["sentiment_binary"] = df["Classification"].apply(
        lambda c: 0 if c in fear_labels else (1 if c in greed_labels else np.nan)
    )

    # ── Direction flag ────────────────────────────────────────────────────────
    # 'direction' contains strings like 'Buy' / 'Sell'; encode to 1 / 0
    df["direction_encoded"] = (df["direction"].str.strip().str.lower() == "buy").astype(int)

    # ── Drop rows with any NaN in core analysis columns ──────────────────────
    core_cols = ["closedPnL", "position_value", "sentiment_encoded"]
    df = df.dropna(subset=core_cols)

    return df


def load_and_prepare(fear_greed_path: str, trades_path: str) -> pd.DataFrame:
    """
    Convenience wrapper: run the full pipeline and return a clean, feature-rich
    DataFrame ready for EDA, statistics, and ML.

    Usage:
        from src.preprocessing import load_and_prepare
        df = load_and_prepare("data/fear_greed.csv", "data/historical_data.csv")
    """
    sentiment_raw, trades_raw = load_raw_data(fear_greed_path, trades_path)
    sentiment = clean_sentiment(sentiment_raw)
    trades    = clean_trades(trades_raw)
    merged    = merge_datasets(sentiment, trades)
    df        = engineer_features(merged)
    print(f"[preprocessing] Final dataset: {df.shape[0]:,} trades × {df.shape[1]} columns")
    print(f"[preprocessing] Sentiment breakdown:\n{df['Classification'].value_counts()}\n")
    return df
