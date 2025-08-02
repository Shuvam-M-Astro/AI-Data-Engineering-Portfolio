# Stock Price Prediction using LSTM Networks

This project implements a deep learning model for predicting stock prices using Long Short-Term Memory (LSTM) networks with comprehensive technical indicators.

## 🎯 Features

- **Real-time Data Fetching**: Uses yfinance to fetch live stock data
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **LSTM Architecture**: Deep learning model for time series prediction
- **Feature Engineering**: Comprehensive data preprocessing
- **Model Evaluation**: Multiple metrics and visualizations
- **Future Prediction**: 30-day price forecasting

## 📊 Technical Indicators Used

- **Moving Averages**: 5-day, 20-day, 50-day
- **RSI (Relative Strength Index)**: Momentum oscillator
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility indicators
- **Volume Analysis**: Volume moving averages and ratios

## 🏗️ Model Architecture

```
LSTM(50) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(25) → Dense(1)
```

## 📈 Usage

```python
from stock_price_prediction import StockPricePredictor

# Initialize predictor
predictor = StockPricePredictor(symbol='AAPL')

# Fetch and prepare data
predictor.fetch_data()
predictor.add_technical_indicators()

# Train model
X_train, X_test, y_train, y_test, data_clean = predictor.prepare_data()
predictor.train_model(X_train, y_train)

# Evaluate and predict
y_pred, y_test_original, mse, mae, rmse = predictor.evaluate_model(X_test, y_test, data_clean)
predictor.plot_results(y_pred, y_test_original, data_clean)

# Predict future prices
future_prices = predictor.predict_future(days=30)
```

## 📋 Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run the prediction**: `python stock_price_prediction.py`

## 📊 Output

The model generates:
- Training history plots
- Prediction vs actual price comparison
- Scatter plot of predictions
- Residual analysis
- Future price predictions

## 🔧 Customization

- **Change Stock Symbol**: Modify the `symbol` parameter
- **Adjust Time Period**: Change `start_date` and `end_date`
- **Modify Lookback**: Adjust the `lookback` parameter for different time windows
- **Add Features**: Include additional technical indicators

## 📈 Performance Metrics

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error  
- **RMSE**: Root Mean Squared Error

## ⚠️ Disclaimer

This project is for educational purposes only. Stock price prediction is inherently difficult and past performance does not guarantee future results. Always do your own research before making investment decisions.

## 🤝 Contributing

Feel free to contribute by:
- Adding new technical indicators
- Improving the model architecture
- Enhancing visualization capabilities
- Adding more evaluation metrics 