
# Stock Price Movement Prediction Using Machine Learning

This project builds a machine learning model to predict stock price movements (up or down) based on historical stock data and technical indicators such as Simple Moving Averages (SMA) and Relative Strength Index (RSI). The project uses Python for data collection, preprocessing, feature engineering, model training, and evaluation. It includes models like Logistic Regression and Random Forest, with hyperparameter tuning for the latter to improve performance.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data](#data)
3. [Features](#features)
4. [Machine Learning Models](#machine-learning-models)
5. [Requirements](#requirements)
6. [How to Run the Project](#how-to-run-the-project)
7. [Results](#results)
8. [Conclusion](#conclusion)

## Project Overview
The goal of this project is to predict whether the closing price of a stock will increase or decrease the following day. To achieve this, we:
- Collect historical stock price data using the Yahoo Finance API (`yfinance` library).
- Engineer new features based on technical indicators like Simple Moving Averages (SMA) and Relative Strength Index (RSI).
- Train and evaluate machine learning models (Logistic Regression and Random Forest) to predict stock price movement.
- Perform hyperparameter tuning on the Random Forest model to optimize performance.

## Data
The historical stock data is fetched from Yahoo Finance using the `yfinance` library. The dataset includes daily stock prices for **Apple Inc. (AAPL)** from January 1, 2015, to January 1, 2023.

### Data Columns:
- **Date**: The date for each observation.
- **Open**: The opening price of the stock.
- **High**: The highest price of the stock for the day.
- **Low**: The lowest price of the stock for the day.
- **Close**: The closing price of the stock.
- **Volume**: The number of shares traded.
  
### Target Variable:
- **Target**: A binary variable indicating whether the closing price increased (`1`) or decreased (`0`) the next day.

## Features
In addition to the raw stock price data, we engineer the following technical indicators:
- **SMA_20**: 20-day Simple Moving Average.
- **SMA_50**: 50-day Simple Moving Average.
- **RSI**: Relative Strength Index (14-day window).

These features are used as inputs to our machine learning models.

## Machine Learning Models

### 1. Logistic Regression
We start by training a **Logistic Regression** model to predict stock price movement. Logistic Regression is a simple yet effective model for binary classification tasks.

### 2. Random Forest
We also train a **Random Forest** model, which is an ensemble method based on decision trees. Random Forest can handle complex datasets and is less prone to overfitting compared to individual decision trees.

### 3. Hyperparameter Tuning
To improve the performance of the Random Forest model, we perform **Grid Search** for hyperparameter tuning. The hyperparameters tuned include:
- `n_estimators`: Number of trees in the forest.
- `max_depth`: Maximum depth of each tree.
- `min_samples_split`: Minimum number of samples required to split a node.
- `min_samples_leaf`: Minimum number of samples required to be at a leaf node.

## Requirements
To run this project, you'll need the following libraries installed:

```bash
pip install pandas numpy matplotlib scikit-learn yfinance
```

## How to Run the Project

1. **Clone the Repository**:
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/stock-price-prediction.git
   ```

2. **Navigate to the Project Directory**:
   ```bash
   cd stock-price-prediction
   ```

3. **Run the Python Script**:
   Run the script to fetch data, preprocess it, train the models, and visualize the results:
   ```bash
   python stock_price_prediction.py
   ```

4. **Expected Output**:
   - The script will output accuracy and performance metrics for both the Logistic Regression and Random Forest models.
   - It will also display plots showing:
     - Stock price movements with technical indicators (SMA_20, SMA_50).
     - Feature importance from the Random Forest model.

## Results
### Model Performance:
- **Logistic Regression**:
  - Baseline model with moderate accuracy.
  
- **Random Forest**:
  - Improved performance compared to Logistic Regression.
  - After hyperparameter tuning, the model achieves even better results.

### Data Visualization:
- Stock price data is visualized along with technical indicators (SMA_20 and SMA_50).
- Feature importance is visualized, showing which technical indicators had the most predictive power in the Random Forest model.

## Conclusion
This project demonstrates how to use machine learning techniques to predict stock price movements based on historical data and technical indicators. The Random Forest model, especially after hyperparameter tuning, performed well and highlighted the importance of various technical features in predicting stock movements.

## Future Work
- **Additional Features**: You can add more technical indicators like Bollinger Bands, MACD, or stochastic oscillators.
- **Multiple Stocks**: Expand the project to predict stock movements for multiple companies or indices.
- **Deep Learning**: Experiment with LSTM (Long Short-Term Memory) networks for time series forecasting.
