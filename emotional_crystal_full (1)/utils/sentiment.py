# ============================================================
# sentiment.py — Emotional Crystal Pro
# ============================================================

import streamlit as st
import requests
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# -------------------------------------------
# FIX: Ensure VADER lexicon exists
# -------------------------------------------
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# Initialize analyzer AFTER ensuring lexicon exists
_analyzer = SentimentIntensityAnalyzer()



# ============================================================
# FETCH NEWS FROM NEWSAPI
# ============================================================

def fetch_news_data(keyword: str) -> pd.DataFrame:
    """
    Fetches news articles from NewsAPI using a keyword.
    Returns DataFrame with columns: timestamp, text, source
    """
    if "NEWS_API_KEY" not in st.secrets:
        st.error("NEWS_API_KEY not set in Streamlit Secrets.")
        return pd.DataFrame()

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": st.secrets["NEWS_API_KEY"],
        "pageSize": 50,
    }

    r = requests.get(url, params=params).json()
    articles = r.get("articles", [])

    rows = []
    for a in articles:
        title = a.get("title", "") or ""
        desc = a.get("description", "") or ""
        txt = f"{title}. {desc}".strip()

        rows.append({
            "timestamp": a.get("publishedAt", ""),
            "text": txt,
            "source": a.get("source", {}).get("name", ""),
        })

    return pd.DataFrame(rows)



# ============================================================
# RUN VADER AND RETURN NEG/NEU/POS/COMPOUND
# ============================================================

def vader_scores(text: str) -> dict:
    """
    Return VADER scores for a given text.
    """
    return _analyzer.polarity_scores(str(text))



# ============================================================
# EXPANDED 20+ EMOTION CLASSIFIER
# ============================================================

def classify_emotion_expanded(row) -> str:
    """
    Full expanded emotion classifier using compound/pos/neg/neu rules.
    Covers:
    - joy, love, pride, hope
    - calm, curiosity, surprise, trust, awe, nostalgia
    - anger, fear, sadness, anxiety, disgust
    - boredom, neutral, mixed
    """

    c = row["compound"]
    pos = row["pos"]
    neg = row["neg"]
    neu = row["neu"]

    # ========================================================
    # Strong Positive Emotions
    # ========================================================
    if c >= 0.75 and pos > 0.60:
        return "joy"
    if c >= 0.55 and pos > 0.45:
        return "love"
    if 0.45 <= c < 0.75 and pos > 0.35:
        return "pride"
    if 0.35 <= c < 0.55 and pos > 0.30:
        return "hope"

    # ========================================================
    # Moderate Positive / Neutral Blend
    # ========================================================
    # calm
    if 0.15 <= c < 0.35 and neu > 0.35:
        return "calm"
    # curiosity
    if 0.05 <= c < 0.25 and (neu > 0.30 or pos > 0.20):
        return "curiosity"
    # surprise
    if pos > 0.25 and abs(c) < 0.20:
        return "surprise"
    # trust
    if 0.10 <= c < 0.35 and pos > 0.25:
        return "trust"
    # awe
    if c >= 0.20 and (pos > 0.20 and neu > 0.20):
        return "awe"
    # nostalgia
    if neu >= 0.40 and (0 <= c < 0.20) and pos > 0.10:
        return "nostalgia"

    # ========================================================
    # Negative Emotions
    # ========================================================
    # anger
    if c <= -0.60 and neg > 0.40:
        return "anger"
    # fear
    if -0.60 < c <= -0.25 and neg > 0.30:
        return "fear"
    # sadness
    if -0.40 < c <= -0.05 and neu > 0.30:
        return "sadness"
    # anxiety
    if -0.15 <= c <= 0.05 and neg > 0.20 and neu < 0.50:
        return "anxiety"
    # disgust
    if neg > 0.35 and c < -0.10:
        return "disgust"

    # ========================================================
    # Neutral / Mixed
    # ========================================================
    if abs(c) < 0.05 and neu > 0.50:
        return "neutral"

    if neu > 0.45 and 0.05 <= abs(c) <= 0.15:
        return "boredom"

    return "mixed"



# ============================================================
# APPLY EMOTION CLASSIFICATION TO DF
# ============================================================

def analyze_sentiment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with a 'text' column and computes:
    - neg, neu, pos, compound using VADER
    - emotion using expanded classifier
    """

    if df.empty:
        return df

    # Compute VADER for each text
    scores = df["text"].apply(vader_scores).tolist()
    score_df = pd.DataFrame(scores)

    df = pd.concat([df.reset_index(drop=True), score_df], axis=1)

    # Run emotion classifier
    df["emotion"] = df.apply(classify_emotion_expanded, axis=1)

    return df
