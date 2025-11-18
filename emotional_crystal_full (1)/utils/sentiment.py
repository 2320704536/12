import streamlit as st
import requests
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

_analyzer = SentimentIntensityAnalyzer()

def fetch_news_data(keyword: str) -> pd.DataFrame:
    return pd.DataFrame()

def analyze_sentiment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df
