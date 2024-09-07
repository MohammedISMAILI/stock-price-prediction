
# stock_price_prediction.py (extended version)

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Step 1: Fetch Stock Data
def fetch_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# Step 2: Data Cleaning and Feature Engineering
def process_data(data):
    # Create simple moving averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # Calculate RSI (Relative Strength Index)
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    data['RSI'] = 100 - (100 / (1 + RS))

    # Create target variable (1 for price increase, 0 for decrease)
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    return data.dropna()

# Step 3: Data Preprocessing
def preprocess_data(data):
    # Select features and target
    X = data[['SMA_20', 'SMA_50', 'RSI']]
    y = data['Target']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Step 4: Model Training - Logistic Regression
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Step 5: Model Training - Random Forest
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Step 6: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred))

# Step 7: Model Improvement - Hyperparameter Tuning for Random Forest
def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    print(f'Best parameters: {grid_search.best_params_}')
    
    return grid_search.best_estimator_

# Step 8: Data Visualization
def visualize_data(data, tuned_rf_model, X_train):
    # Visualize Stock Price with Moving Averages
    plt.figure(figsize=(10,6))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['SMA_20'], label='SMA 20', linestyle='--')
    plt.plot(data['SMA_50'], label='SMA 50', linestyle='--')
    plt.title('Stock Price with Moving Averages')
    plt.legend()
    plt.show()

    # Visualize Feature Importance (for Random Forest)
    plt.figure(figsize=(8,6))
    plt.barh(X_train.columns, tuned_rf_model.feature_importances_)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Stock Price Prediction (Tuned Model)')
    plt.show()

# Main function
def main():
    stock_symbol = 'AAPL'  # Example stock: Apple
    start_date = '2015-01-01'
    end_date = '2023-01-01'
    
    # Step 1: Fetch and process stock data
    stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
    processed_data = process_data(stock_data)
    
    # Step 2: Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(processed_data)
    
    # Step 3: Train Logistic Regression model and evaluate
    print("Logistic Regression Model:")
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test)
    
    # Step 4: Train Random Forest model and evaluate
    print("Random Forest Model:")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)
    
    # Step 5: Improve Random Forest model with hyperparameter tuning
    print("Tuning Random Forest:")
    tuned_rf_model = tune_random_forest(X_train, y_train)
    evaluate_model(tuned_rf_model, X_test, y_test)
    
    # Step 6: Visualize data and feature importance
    visualize_data(processed_data, tuned_rf_model, X_train)

if __name__ == '__main__':
    main()
