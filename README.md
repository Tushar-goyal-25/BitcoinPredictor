# Bitcoin Price Predictor Dashboard

This project is a Bitcoin price trend prediction dashboard that leverages LSTM and XGBoost models to forecast short-term Bitcoin market movements. The interactive dashboard, built using Streamlit, provides visual insights into price trends, sentiment, and model predictions. It is designed especially for novice traders to gain a better understanding of potential market directions.

## Features

- Bitcoin Price Prediction using:
  - LSTM Neural Networks for regression
  - XGBoost Classifier for trend classification (up/down)
- Sentiment Analysis from:
  - Wikipedia Bitcoin page edit history
  - Bitcoin-related news articles
- Interactive Streamlit Dashboard
- Visualizations for:
  - Predicted vs actual price trends
  - Confusion matrix for evaluation
  - Feature importance for transparency

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Tushar-goyal-25/BitcoinPredictor.git
cd BitcoinPredictor
````

### 2. Install Requirements

It is recommended to use a virtual environment. Install required dependencies using:

```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard

Run the main Streamlit dashboard using the following command:

```bash
streamlit run dashboard.py
```


## Dashboard Capabilities

* Visualize Bitcoin closing prices with 7-day and 30-day moving averages
* Select prediction horizons (Next Day, Week, Month, Year)
* Display model output as directional movement (UP or DOWN)
* View confusion matrix and model precision
* Explore feature importance and sentiment impacts

## Data Sources

* Yahoo Finance (for Bitcoin price history)
* Wikipedia Dumps (Bitcoin article revisions)
* Bitcoin-related news dataset (from Kaggle)

## Future Work

* Integrate real-time data updates
* Tune sentiment weighting in model training
* Extend prediction horizons with confidence intervals

## Author

Tushar Goyal
Queen Mary University of London
Student ID: 220646213


