# import mwclient 
# import time 
# from transformers import pipeline
# from statistics import mean
# import pandas as pd
# from datetime import datetime
# import feedparser

# def get_bitcoin_news():
#     rss_url = "https://news.google.com/rss/search?q=Bitcoin&hl=en-US&gl=US&ceid=US:en"
#     feed = feedparser.parse(rss_url)
    
#     articles = []
#     for entry in feed.entries:
#         published = entry.published  # Example: 'Sun, 28 Apr 2024 15:30:00 GMT'
#         title = entry.title
#         summary = entry.summary  # Sometimes empty
        
#         articles.append({
#             "date": pd.to_datetime(published).date(),
#             "text": title + " " + summary
#         })
#     return pd.DataFrame(articles)



# site = mwclient.Site("en.wikipedia.org")
# page = site.pages["Bitcoin"]
# revs = list(page.revisions())
# # print(revs[0])
# revs = sorted(revs, key=lambda rev: rev["timestamp"])
# sentiment_pipeline = pipeline("sentiment-analysis")

# def find_sentiment(text):
#     sent = sentiment_pipeline([text[:250]])[0]
#     score = sent["score"]
#     if sent["label"] == "NEGATIVE":
#         score *= -1
#     return score

# edits = {}

# for rev in revs:
#     date = time.strftime("%Y-%m-%d", rev["timestamp"])

#     if date not in edits:
#         edits[date] = dict(sentiments = list(), edit_count=0)
#     edits[date]["edit_count"] += 1
#     comment = rev.get("comment", "")
#     edits[date]["sentiments"].append(find_sentiment(comment))

# for key in edits:
#     if len(edits[key]["sentiments"]) > 0:
#         edits[key]["sentiment"] = mean(edits[key]["sentiments"])
#         edits[key]["neg_sentiment"] = len([s for s in edits[key]["sentiments"] if s < 0 ]) / len(edits[key]["sentiments"])
#     else:
#         edits[key]["sentiment"] = 0
#         edits[key]["neg_sentiment"] = 0

#     del edits[key]["sentiments"]

# # Pull live news
# news_df = get_bitcoin_news()

# # Find sentiment
# news_df["sentiment"] = news_df["text"].apply(lambda x: find_sentiment(x))

# # Aggregate sentiment by date

# # Rolling average and save


# edits_df = pd.DataFrame.from_dict(edits, orient="index")
# edits_df.index = pd.to_datetime(edits_df.index)
# dates = pd.date_range(start="2009-03-08", end=datetime.today())
# edits_df = edits_df.reindex(dates, fill_value=0)
# daily_news_sentiment = news_df.groupby(news_df["date"])["sentiment"].mean()
# daily_news_sentiment = daily_news_sentiment.reindex(dates, fill_value=0)
# # Merge into Wikipedia edits dataframe
# edits_df["article_sentiment"] = daily_news_sentiment.values
# rolling_edits = edits_df.rolling(30).mean()
# rolling_edits = rolling_edits.dropna()
# rolling_edits.to_csv("wikipedia_and_news_edits.csv")

import mwclient 
import time 
from transformers import pipeline
from statistics import mean
import pandas as pd
from datetime import datetime

# Set up Wikipedia site and page
site = mwclient.Site("en.wikipedia.org")
page = site.pages["Bitcoin"]
revs = list(page.revisions())
revs = sorted(revs, key=lambda rev: rev["timestamp"])

# Set up Huggingface Sentiment Analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def find_sentiment(text):
    sent = sentiment_pipeline([text[:250]])[0]
    score = sent["score"]
    if sent["label"] == "NEGATIVE":
        score *= -1
    return score

# Step 1: Wikipedia edits sentiment
edits = {}

for rev in revs:
    date = time.strftime("%Y-%m-%d", rev["timestamp"])
    if date not in edits:
        edits[date] = dict(sentiments=[], edit_count=0)
    edits[date]["edit_count"] += 1
    comment = rev.get("comment", "")
    edits[date]["sentiments"].append(find_sentiment(comment))

for key in edits:
    if len(edits[key]["sentiments"]) > 0:
        edits[key]["sentiment"] = mean(edits[key]["sentiments"])
        edits[key]["neg_sentiment"] = len([s for s in edits[key]["sentiments"] if s < 0]) / len(edits[key]["sentiments"])
    else:
        edits[key]["sentiment"] = 0
        edits[key]["neg_sentiment"] = 0
    del edits[key]["sentiments"]

# Create DataFrame from Wikipedia data
edits_df = pd.DataFrame.from_dict(edits, orient="index")
edits_df.index = pd.to_datetime(edits_df.index)
dates = pd.date_range(start="2009-03-08", end=datetime.today())
edits_df = edits_df.reindex(dates, fill_value=0)
edits_df = edits_df.astype(float)

# Step 2: Load Bitcoin news sentiments from your CSV
news_df = pd.read_csv("/Users/tushar/Desktop/UNI/FinalYEARProject/ytcode/bitcoin_sentiments_21_24.csv")  # Adjust path if needed
news_df["Date"] = pd.to_datetime(news_df["Date"])

# Aggregate daily sentiment
daily_news_sentiment = news_df.groupby(news_df["Date"].dt.date)["Sentiments"].mean()

# Step 3: Create full range and merge
news_sentiment_full = pd.Series(0, index=dates.date)
news_sentiment_full.update(daily_news_sentiment)

# Add to Wikipedia edits dataframe
edits_df["article_sentiment"] = news_sentiment_full.values

# Step 4: Rolling 30-day average
rolling_edits = edits_df.rolling(30).mean()
rolling_edits = rolling_edits.dropna()

# Step 5: Save to CSV
rolling_edits.to_csv("wikipedia_and_news_edits.csv")
