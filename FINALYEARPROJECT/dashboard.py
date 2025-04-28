import yfinance as yf
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from xgboost import XGBClassifier
<<<<<<< Updated upstream
import os
import streamlit as st


# Load Data
@st.cache_data
def load_data():
    btc_ticker = yf.Ticker("BTC-GBP")
    btc = btc_ticker.history(period="max")
    btc.index = pd.to_datetime(btc.index)
    del btc["Dividends"]
    del btc["Stock Splits"]
    btc.columns = [c.lower() for c in btc.columns]
    return btc

# Feature Engineering
def add_features(btc):
    btc["return_1d"] = btc["close"].pct_change()
    btc["return_7d"] = btc["close"].pct_change(7)
    btc["volatility_7d"] = btc["return_1d"].rolling(7).std()
    btc["momentum_7d"] = btc["close"] / btc["close"].shift(7)
    btc["sma_7"] = btc["close"].rolling(window=7).mean()
    btc["sma_30"] = btc["close"].rolling(window=30).mean()
    btc["close_to_sma_7"] = btc["close"] / btc["sma_7"]
    btc.fillna(0, inplace=True)
    return btc

# Rolling Average Feature Engineering
def compute_rolling(btc):
    horizons = [2, 7, 60, 365]
    new_predictors = [
        "close", "sentiment", "neg_sentiment", "article_sentiment",
        "return_1d", "return_7d", "volatility_7d", "momentum_7d",
        "sma_7", "sma_30", "close_to_sma_7"
    ]

    for horizon in horizons:
        rolling_averages = btc.rolling(horizon, min_periods=1).mean()
        ratio_column = f"close_ratio_{horizon}"
        btc[ratio_column] = btc["close"] / rolling_averages["close"]
        edit_column = f"edit_{horizon}"
        btc[edit_column] = rolling_averages["edit_count"]
        article_sentiment_column = f"article_sentiment_{horizon}"
        btc[article_sentiment_column] = rolling_averages["article_sentiment"]
        return_1d_col = f"return_1d_{horizon}"
        return_7d_col = f"return_7d_{horizon}"
        volatility_7d_col = f"volatility_7d_{horizon}"
        momentum_7d_col = f"momentum_7d_{horizon}"
        close_to_sma_7_col = f"close_to_sma_7_{horizon}"

        btc[return_1d_col] = rolling_averages["return_1d"]
        btc[return_7d_col] = rolling_averages["return_7d"]
        btc[volatility_7d_col] = rolling_averages["volatility_7d"]
        btc[momentum_7d_col] = rolling_averages["momentum_7d"]
        btc[close_to_sma_7_col] = rolling_averages["close_to_sma_7"]

        rolling = btc.rolling(horizon, closed="left", min_periods=1).mean()
        trend_column = f"trend_{horizon}"
        btc[trend_column] = rolling["target"]

        new_predictors += [
            ratio_column, trend_column, edit_column, article_sentiment_column,
            return_1d_col, return_7d_col, volatility_7d_col, momentum_7d_col, close_to_sma_7_col
        ]

    return btc, new_predictors


# Prediction & Backtesting
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["target"], eval_set=[(test[predictors], test["target"])], verbose=False)
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="predictions")
    return pd.concat([test["target"], preds], axis=1)

def backtest(data, model, predictors, start=1095, step=150):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i]
        test = data.iloc[i:i+step]
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

# Main Dashboard
def main():
    
    st.markdown("# 📈 Bitcoin Price Prediction Dashboard")

    btc = load_data()
    st.subheader("Live Bitcoin Price Refresh")
    if st.button("🔄 Refresh Latest Price"):
        btc_latest = yf.download("BTC-GBP", period="1d", interval="1m")
        latest_price = float(btc_latest["Close"].dropna().values[-1])
        st.success(f"Latest Bitcoin Price: £{latest_price:,.2f}")
    else:
        st.info("Click the button to fetch the live price.")

    # Merge with Sentiment Data (Optional)
    wiki_path = "../wikipedia_and_news_edits.csv"
    if os.path.exists(wiki_path):
        wiki = pd.read_csv(wiki_path, index_col=0, parse_dates=True)
        btc.index = btc.index.tz_localize(None)
        if "edit_count" in wiki.columns:
            btc = btc.merge(wiki, left_index=True, right_index=True)
        else:
            st.warning("⚠️ Sentiment data file loaded but missing expected columns like 'edit_count'.")
    else:
        st.warning("⚠️ Sentiment data file 'wikipedia_and_news_edits.csv' not found.")

    btc = add_features(btc)

    # Target Variable
    btc["tomorrow"] = btc["close"].shift(-1)
    btc["target"] = (btc["tomorrow"] > btc["close"]).astype(int)
    btc.fillna(0, inplace=True)

    btc, predictors = compute_rolling(btc)


    model = XGBClassifier(random_state=1, learning_rate=0.01, n_estimators=500, eval_metric="aucpr")

    # Show Price Chart
    st.subheader("Bitcoin Closing Price Over Time")
    st.line_chart(btc[["close", "sma_7", "sma_30"]])

    # Prediction Options
    st.subheader("Predict Bitcoin Movement")
    option = st.selectbox(
        "Choose prediction horizon:",
        ("Next Day", "Next Week", "Next Month", "Next Year")
    )

    horizon_mapping = {
        "Next Day": 1,
        "Next Week": 7,
        "Next Month": 30,
        "Next Year": 365
    }
    horizon = horizon_mapping[option]

    # Train and Predict
    train = btc.iloc[:-horizon]
    test = btc.iloc[-horizon:]

    model.fit(train[predictors], train["target"])
    preds = model.predict(test[predictors])

    direction = "🔺 UP" if preds[-1] == 1 else "🔻 DOWN"

    # Show Prediction
    st.metric(label=f"Prediction for {option}:", value=direction)

    # Backtest
    with st.spinner("Calculating model precision..."):
        backtest_preds = backtest(btc, model, predictors)
        score = precision_score(backtest_preds["target"], backtest_preds["predictions"])

    st.success(f"Model Precision: {score:.2f}")

if __name__ == "__main__":
    main()
