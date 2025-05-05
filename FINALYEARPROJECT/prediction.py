import yfinance as yf
import os 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
btc_ticker = yf.Ticker("BTC-GBP")
btc = btc_ticker.history(period = "max")
btc.index= pd.to_datetime(btc.index)
del btc["Dividends"]
del btc["Stock Splits"]

btc.columns = [c.lower() for c in btc.columns]
btc.plot.line(y="close", use_index=True)
def add_features(btc):
    # Daily Returns
    btc["return_1d"] = btc["close"].pct_change()

    # 7-day Return (weekly trend)
    btc["return_7d"] = btc["close"].pct_change(7)

    # Volatility (standard deviation of returns)
    btc["volatility_7d"] = btc["return_1d"].rolling(7).std()

    # Momentum: Is price trending upward over last 7 days?
    btc["momentum_7d"] = btc["close"] / btc["close"].shift(7)

    # Moving Averages (Simple 7-day and 30-day)
    btc["sma_7"] = btc["close"].rolling(7).mean()
    btc["sma_30"] = btc["close"].rolling(30).mean()

    # Price relative to 7-day SMA
    btc["close_to_sma_7"] = btc["close"] / btc["sma_7"]

    btc.fillna(0, inplace=True)
    return btc


wiki = pd.read_csv("/Users/tushar/Desktop/UNI/FinalYEARProject/wikipedia_and_news_edits.csv", index_col=0, parse_dates=True)
btc.index = btc.index.tz_localize(None)
btc = btc.merge(wiki, left_index=True, right_index = True) 
btc = add_features(btc)

btc["tomorrow"] = btc["close"].shift(-1)
btc["target"]= (btc["tomorrow"] > btc["close"]).astype(int)
btc["target"].value_counts()
btc.fillna(0, inplace=True)
model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)

train = btc.iloc[:-200]
test = btc[-200:]
predictors = ["close", "volume", "open", "high", "low", "edit_count", "sentiment", "neg_sentiment","article_sentiment","return_1d", "return_7d", "volatility_7d", "momentum_7d",
    "sma_7", "sma_30", "close_to_sma_7"]
model.fit(train[predictors], train["target"])
preds = model.predict(test[predictors])
preds = pd.Series(preds, index = test.index)
print(precision_score(test["target"], preds))



def predict(train,test,predictors,model): #prediction code 
    model.fit(train[predictors], train["target"],eval_set=[(test[predictors], test["target"])],verbose=False)
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index = test.index, name="predictions")    
    precision_score(test["target"], preds)
    combine = pd.concat([test["target"], preds],axis=1)
    return combine

def backtest(data, model, predictors, start=1095, step=150): #backtest code
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:i+step].copy()
        predictions = predict(train,test,predictors,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


model = XGBClassifier(random_state=1, learning_rate =0.01, n_estimators= 500,eval_metric="aucpr" ) #model defined 
predictions = backtest(btc,model,predictors)



print(precision_score(predictions["target"], predictions["predictions"]))

def compute_rolling(btc):
    horizons = [2,7,60,365]
    new_predictors = ["close","sentiment", "neg_sentiment", "article_sentiment","return_1d", "return_7d", "volatility_7d", "momentum_7d",
                      "sma_7", "sma_30", "close_to_sma_7"]

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

        rolling = btc.rolling(horizon, closed="left" , min_periods=1).mean()
        trend_column = f"trend_{horizon}"
        btc[trend_column] = rolling["target"]

        new_predictors += [ratio_column, trend_column, edit_column, article_sentiment_column,return_1d_col, return_7d_col, volatility_7d_col, momentum_7d_col, close_to_sma_7_col]
    return btc, new_predictors

btc,new_predictors = compute_rolling(btc.copy())
predictions = backtest(btc, model, new_predictors)
print(precision_score(predictions["target"], predictions["predictions"]))


