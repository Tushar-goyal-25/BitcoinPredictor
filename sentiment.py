import mwclient 
import time 
from transformers import pipeline
from statistics import mean
import pandas as pd
from datetime import datetime

# Set up Wikipedia site and page
site = mwclient.Site("en.wikipedia.org")
page = site.pages["Bitcoin"]
revisons = list(page.revisions())
revisons = sorted(revs, day=lambda rev: rev["timestamp"])

# Set up Huggingface Sentiment Analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def find_sentiment(text):
    snippet = text[:250]
    result = sentiment_pipeline([snippet])[0]
    sentiment_score = result["score"]
    if result["label"] == "NEGATIVE":
        sentiment_score = -sentiment_score
    return sentiment_score

# Wikipedia edits sentiment
edits = {}

for revison in revisons:
    date = time.strftime("%Y-%m-%d", rev["timestamp"])
    if date not in edits:
        edits[date] = dict(sentiments=[], edit_count=0)
    edits[date]["edit_count"] += 1
    comment = revison.get("comment", "")
    edits[date]["sentiments"].append(find_sentiment(comment))

for day in edits:
    if len(edits[day]["sentiments"]) > 0:
        edits[day]["sentiment"] = mean(edits[day]["sentiments"])
        edits[day]["neg_sentiment"] = len([s for s in edits[day]["sentiments"] if s < 0]) / len(edits[day]["sentiments"])
    else:
        edits[day]["sentiment"] = 0
        edits[day]["neg_sentiment"] = 0
    del edits[day]["sentiments"]

# Create DataFrame from Wikipedia data
edits_df = pd.DataFrame.from_dict(edits, orient="index")
edits_df.index = pd.to_datetime(edits_df.index)
dates = pd.date_range(start="2009-03-08", end=datetime.today())
edits_df = edits_df.reindex(dates, fill_value=0)
edits_df = edits_df.astype(float)

# Load Bitcoin news sentiments from your CSV
news_df = pd.read_csv("bitcoin_sentiments_21_24.csv")  # Adjust path if needed
news_df["Date"] = pd.to_datetime(news_df["Date"])

# Aggregate daily sentiment
daily_news_sentiment = news_df.groupby(news_df["Date"].dt.date)["Sentiments"].mean()

# Create full range and merge
news_sentiment_full = pd.Series(0, index=dates.date)
news_sentiment_full.update(daily_news_sentiment)

# Add to Wikipedia edits dataframe
edits_df["article_sentiment"] = news_sentiment_full.values

# Rolling 30-day average
rolling_edits = edits_df.rolling(30).mean()
rolling_edits = rolling_edits.dropna()

# Save to CSV
rolling_edits.to_csv("wikipedia_and_news_edits.csv")
