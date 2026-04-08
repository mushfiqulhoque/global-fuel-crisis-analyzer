"""
sentiment.py — News headline sentiment analysis (Bonus module).

Fetches headlines related to oil prices / fuel crises from the GNews API
(free tier) and scores them using TextBlob's lexicon-based analyser plus
an optional HuggingFace transformer (FinBERT / distilbert-finance).

Outputs a DataFrame: date | headline | source | polarity | subjectivity | label
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from loguru import logger
from textblob import TextBlob

try:
    from transformers import pipeline as hf_pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("transformers not installed — HuggingFace sentiment disabled.")

GNEWS_BASE = "https://gnews.io/api/v4/search"

# ---------------------------------------------------------------------------
# TextBlob-based scorer (always available, no GPU needed)
# ---------------------------------------------------------------------------

def _textblob_score(text: str) -> dict:
    """Return polarity (−1..+1) and subjectivity (0..1) via TextBlob."""
    blob = TextBlob(text)
    polarity     = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)
    label = "positive" if polarity > 0.05 else ("negative" if polarity < -0.05 else "neutral")
    return {"polarity": polarity, "subjectivity": subjectivity, "label": label}


# ---------------------------------------------------------------------------
# HuggingFace scorer (optional)
# ---------------------------------------------------------------------------

_hf_scorer = None  # lazy-loaded


def _get_hf_scorer():
    global _hf_scorer
    if _hf_scorer is None:
        logger.info("Loading HuggingFace sentiment model (distilbert) …")
        _hf_scorer = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
    return _hf_scorer


def _hf_score(text: str) -> dict:
    scorer = _get_hf_scorer()
    result = scorer(text[:512])[0]
    polarity = result["score"] if result["label"] == "POSITIVE" else -result["score"]
    return {
        "polarity": round(polarity, 4),
        "label": result["label"].lower(),
        "confidence": round(result["score"], 4),
    }


# ---------------------------------------------------------------------------
# Headline fetcher (GNews)
# ---------------------------------------------------------------------------

def fetch_headlines(
    query: str = "oil price fuel crisis",
    api_key: Optional[str] = None,
    days_back: int = 30,
    max_results: int = 50,
) -> pd.DataFrame:
    """
    Fetch recent news headlines from GNews API.

    Falls back to a hardcoded sample dataset if no API key is provided,
    so the project still runs end-to-end without external news credentials.
    """
    if not api_key:
        logger.warning("No GNEWS_API_KEY provided — using synthetic headline samples.")
        return _synthetic_headlines()

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "q":        query,
        "lang":     "en",
        "from":     from_date,
        "max":      min(max_results, 100),
        "apikey":   api_key,
        "sortby":   "publishedAt",
    }
    resp = requests.get(GNEWS_BASE, params=params, timeout=20)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])

    rows = []
    for a in articles:
        rows.append({
            "date":      pd.to_datetime(a.get("publishedAt", "")).date(),
            "headline":  a.get("title", ""),
            "source":    a.get("source", {}).get("name", ""),
            "url":       a.get("url", ""),
        })
    return pd.DataFrame(rows)


def _synthetic_headlines() -> pd.DataFrame:
    """Return a curated set of synthetic headlines relative to today's date."""
    base = datetime.utcnow()  # ← FIXED: was hardcoded datetime(2024, 1, 1)
    data = [
        (0,  "Oil prices surge as OPEC+ announces surprise production cut",     "Reuters"),
        (2,  "Brent crude slips amid demand concerns from China slowdown",       "Bloomberg"),
        (4,  "Energy markets volatile as US-Iran tensions escalate in 2026",    "FT"),
        (6,  "Fuel costs ease slightly across Europe after OPEC+ output hike",  "AP"),
        (8,  "OPEC holds output steady; prices ease on demand concerns",         "WSJ"),
        (10, "Global oil supply rebounds as non-OPEC output rises",             "Reuters"),
        (12, "Crude prices fall sharply on weaker Chinese demand data",          "Bloomberg"),
        (14, "IEA warns of tight oil market through mid-2026",                  "IEA"),
        (17, "US strategic petroleum reserve release helps stabilise prices",    "CNBC"),
        (20, "Geopolitical tensions push crude to two-month high",              "Reuters"),
        (22, "Oil demand outlook cut by analysts amid recession worries",        "Bloomberg"),
        (25, "Pipeline disruption in key corridor spikes European gas prices",  "FT"),
        (27, "Saudi Arabia pledges output stability at OPEC ministerial",       "AFP"),
        (30, "Markets calm as ceasefire talks progress; oil falls 3%",          "Reuters"),
    ]
    rows = [
        {"date": (base - timedelta(days=d)).date(), "headline": h, "source": s}
        for d, h, s in data
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------

def score_headlines(
    df: pd.DataFrame,
    use_hf: bool = False,
) -> pd.DataFrame:
    """
    Add sentiment scores to a headlines DataFrame.

    Parameters
    ----------
    df     : DataFrame with at least a 'headline' column
    use_hf : if True (and transformers installed), use HuggingFace in addition

    Returns
    -------
    DataFrame with added columns: polarity, subjectivity, label
    """
    df = df.copy()
    records = []
    for _, row in df.iterrows():
        headline = str(row.get("headline", ""))
        scores   = _textblob_score(headline)
        if use_hf and HF_AVAILABLE:
            try:
                hf = _hf_score(headline)
                scores["hf_polarity"]   = hf["polarity"]
                scores["hf_label"]      = hf["label"]
                scores["hf_confidence"] = hf.get("confidence", 0.0)
                # Blend: 50/50 average
                scores["blended_polarity"] = round(
                    (scores["polarity"] + hf["polarity"]) / 2, 4
                )
            except Exception as exc:
                logger.debug(f"HF scorer error: {exc}")
        records.append(scores)

    scores_df = pd.DataFrame(records)
    result    = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
    logger.success(f"Scored {len(result)} headlines. "
                   f"Positive: {(result['label']=='positive').sum()}  "
                   f"Neutral: {(result['label']=='neutral').sum()}  "
                   f"Negative: {(result['label']=='negative').sum()}")
    return result


def aggregate_daily_sentiment(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate headline-level scores to daily sentiment.
    Returns DataFrame indexed by date with mean polarity and dominant label.
    """
    scored_df["date"] = pd.to_datetime(scored_df["date"])
    daily = (
        scored_df
        .groupby("date")
        .agg(
            mean_polarity=("polarity", "mean"),
            headline_count=("headline", "count"),
            dominant_label=("label", lambda x: x.mode()[0]),
        )
        .reset_index()
        .set_index("date")
        .sort_index()
    )
    return daily


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    api_key = os.getenv("GNEWS_API_KEY", "")
    headlines = fetch_headlines(api_key=api_key, days_back=30)
    scored    = score_headlines(headlines, use_hf=False)
    daily     = aggregate_daily_sentiment(scored)
    print(scored[["date", "headline", "polarity", "label"]].to_string(index=False))
    print("\nDaily sentiment:")
    print(daily)
