# Weather Forecasting

Lag-based time series forecasting for meteorological data using RandomForest.

## Usage

```bash
pip install -r requirements.txt
python weather_forecasting.py --data-csv path/to/weather.csv --target-column temperature --date-column date --n-lags 24 --test-size 168
```

If `--data-csv` is omitted, synthetic data is generated.

CSV should contain at least two columns: `date`, `value` (or set `--target-column`).
