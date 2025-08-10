# Energy Consumption Forecasting

Lag-based time series forecasting for power usage using RandomForest.

## Usage

```bash
pip install -r requirements.txt
python energy_consumption.py --data-csv path/to/energy.csv --target-column load --date-column date --n-lags 48 --test-size 336
```

If `--data-csv` is omitted, synthetic data is generated.

CSV should contain at least two columns: `date`, `value` (or set `--target-column`).
