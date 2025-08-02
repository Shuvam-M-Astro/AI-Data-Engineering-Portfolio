"""
Stock Price Prediction using LSTM Networks
=========================================

This project implements a deep learning model for predicting stock prices
using Long Short-Term Memory (LSTM) networks with time series data.

Features:
- Data preprocessing and feature engineering
- LSTM model architecture
- Technical indicators integration
- Model evaluation and visualization
- Real-time prediction capabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class StockPricePredictor:
    def __init__(self, symbol='AAPL', start_date='2020-01-01', end_date='2024-01-01'):
        """
        Initialize the stock price predictor.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date for data collection
            end_date (str): End date for data collection
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.scaler = MinMaxScaler()
        self.model = None
        self.history = None
        
    def fetch_data(self):
        """Fetch stock data using yfinance."""
        print(f"Fetching data for {self.symbol}...")
        self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        print(f"Data shape: {self.data.shape}")
        return self.data
    
    def add_technical_indicators(self):
        """Add technical indicators to the dataset."""
        # Moving averages
        self.data['MA5'] = self.data['Close'].rolling(window=5).mean()
        self.data['MA20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA50'] = self.data['Close'].rolling(window=50).mean()
        
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.data['Close'].ewm(span=12).mean()
        exp2 = self.data['Close'].ewm(span=26).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        self.data['BB_Middle'] = self.data['Close'].rolling(window=20).mean()
        bb_std = self.data['Close'].rolling(window=20).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (bb_std * 2)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
        
        return self.data
    
    def prepare_data(self, lookback=60):
        """
        Prepare data for LSTM model.
        
        Args:
            lookback (int): Number of time steps to look back
        """
        # Select features
        features = ['Close', 'Volume', 'MA5', 'MA20', 'MA50', 'RSI', 
                   'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 
                   'Volume_Ratio']
        
        # Remove NaN values
        data_clean = self.data[features].dropna()
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data_clean)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0])  # Predict Close price
        
        X, y = np.array(X), np.array(y)
        
        # Split into train and test sets
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        return X_train, X_test, y_train, y_test, data_clean
    
    def build_model(self, lookback, n_features):
        """
        Build LSTM model.
        
        Args:
            lookback (int): Number of time steps
            n_features (int): Number of features
        """
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(lookback, n_features)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def train_model(self, X_train, y_train, epochs=50, batch_size=32):
        """Train the LSTM model."""
        self.model = self.build_model(X_train.shape[1], X_train.shape[2])
        
        print("Training model...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        return self.model
    
    def evaluate_model(self, X_test, y_test, data_clean):
        """Evaluate the model and create predictions."""
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Inverse transform predictions
        y_pred_original = self.scaler.inverse_transform(
            np.concatenate([y_pred, np.zeros((len(y_pred), data_clean.shape[1]-1))], axis=1)
        )[:, 0]
        
        y_test_original = self.scaler.inverse_transform(
            np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), data_clean.shape[1]-1))], axis=1)
        )[:, 0]
        
        # Calculate metrics
        mse = mean_squared_error(y_test_original, y_pred_original)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        
        return y_pred_original, y_test_original, mse, mae, rmse
    
    def plot_results(self, y_pred, y_test, data_clean):
        """Plot the results."""
        # Create date index for test set
        test_dates = data_clean.index[-len(y_test):]
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Training history
        plt.subplot(2, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot 2: Predictions vs Actual
        plt.subplot(2, 2, 2)
        plt.plot(test_dates, y_test, label='Actual', color='blue')
        plt.plot(test_dates, y_pred, label='Predicted', color='red')
        plt.title('Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Plot 3: Scatter plot
        plt.subplot(2, 2, 3)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Prediction vs Actual')
        
        # Plot 4: Residuals
        plt.subplot(2, 2, 4)
        residuals = y_test - y_pred
        plt.plot(test_dates, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals')
        plt.xlabel('Date')
        plt.ylabel('Residual')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def predict_future(self, days=30):
        """Predict future stock prices."""
        # Get the last sequence
        last_sequence = self.data[['Close', 'Volume', 'MA5', 'MA20', 'MA50', 'RSI', 
                                  'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 
                                  'Volume_Ratio']].tail(60).values
        
        # Scale the sequence
        last_sequence_scaled = self.scaler.transform(last_sequence)
        
        # Make prediction
        future_predictions = []
        current_sequence = last_sequence_scaled.copy()
        
        for _ in range(days):
            # Predict next value
            next_pred = self.model.predict(current_sequence.reshape(1, 60, 11))
            future_predictions.append(next_pred[0, 0])
            
            # Update sequence (shift and add prediction)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = next_pred[0]
        
        # Inverse transform predictions
        future_predictions_original = self.scaler.inverse_transform(
            np.concatenate([np.array(future_predictions).reshape(-1, 1), 
                           np.zeros((len(future_predictions), 10))], axis=1)
        )[:, 0]
        
        return future_predictions_original

def main():
    """Main function to run the stock price prediction."""
    # Initialize predictor
    predictor = StockPricePredictor(symbol='AAPL')
    
    # Fetch and prepare data
    predictor.fetch_data()
    predictor.add_technical_indicators()
    
    # Prepare data for training
    X_train, X_test, y_train, y_test, data_clean = predictor.prepare_data(lookback=60)
    
    # Train model
    predictor.train_model(X_train, y_train, epochs=50)
    
    # Evaluate model
    y_pred, y_test_original, mse, mae, rmse = predictor.evaluate_model(X_test, y_test, data_clean)
    
    # Plot results
    predictor.plot_results(y_pred, y_test_original, data_clean)
    
    # Predict future prices
    future_prices = predictor.predict_future(days=30)
    print(f"Predicted prices for next 30 days: {future_prices}")

if __name__ == "__main__":
    main() 