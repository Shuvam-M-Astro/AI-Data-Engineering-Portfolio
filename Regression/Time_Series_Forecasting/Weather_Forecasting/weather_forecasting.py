import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass
class ForecastConfig:
    data_csv: Optional[str] = None
    target_column: str = "value"
    date_column: str = "date"
    n_lags: int = 24
    test_size: int = 168  # last N points (one week for hourly data)
    random_state: int = 42
    n_estimators: int = 300
    max_depth: Optional[int] = None


def generate_synthetic_weather(n_points: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2020-01-01", periods=n_points, freq="H")

    trend = np.linspace(0, 5, n_points)
    daily_seasonality = 10 * np.sin(2 * np.pi * np.arange(n_points) / 24)
    yearly_seasonality = 5 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 365))
    noise = rng.normal(0, 1.5, n_points)

    value = 20 + trend + daily_seasonality + yearly_seasonality + noise

    return pd.DataFrame({"date": dates, "value": value})


def load_data(config: ForecastConfig) -> pd.DataFrame:
    if config.data_csv and Path(config.data_csv).exists():
        df = pd.read_csv(config.data_csv)
    else:
        df = generate_synthetic_weather()

    if config.date_column in df.columns:
        df[config.date_column] = pd.to_datetime(df[config.date_column])
        df = df.sort_values(config.date_column).reset_index(drop=True)
    else:
        raise ValueError(f"Date column '{config.date_column}' not found in data")

    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' not found in data")

    return df[[config.date_column, config.target_column]].copy()


def create_lag_features(series: pd.Series, n_lags: int) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df.dropna(inplace=True)
    return df


def train_test_split_time(df: pd.DataFrame, test_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if test_size <= 0 or test_size >= len(df):
        raise ValueError("test_size must be between 1 and len(df)-1")
    return df.iloc[:-test_size, :].copy(), df.iloc[-test_size:, :].copy()


def train_and_evaluate(config: ForecastConfig) -> None:
    df = load_data(config)

    lagged = create_lag_features(df[config.target_column], n_lags=config.n_lags)

    train_df, test_df = train_test_split_time(lagged, test_size=config.test_size)

    X_train = train_df.drop(columns=["y"]).values
    y_train = train_df["y"].values
    X_test = test_df.drop(columns=["y"]).values
    y_test = test_df["y"].values

    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Weather Forecasting Results")
    print("-" * 32)
    print(f"Test size: {len(y_test)}")
    print(f"MAE: {mae:.3f}")
    print(f"R2: {r2:.3f}")


def parse_args() -> ForecastConfig:
    parser = argparse.ArgumentParser(
        description="Weather forecasting using lag features and RandomForest"
    )
    parser.add_argument(
        "--data-csv", dest="data_csv", type=str, default=None, help="Path to input CSV"
    )
    parser.add_argument(
        "--target-column",
        dest="target_column",
        type=str,
        default="value",
        help="Target column name",
    )
    parser.add_argument(
        "--date-column", dest="date_column", type=str, default="date", help="Date column name"
    )
    parser.add_argument(
        "--n-lags", dest="n_lags", type=int, default=24, help="Number of lag features"
    )
    parser.add_argument(
        "--test-size", dest="test_size", type=int, default=168, help="Test set size (last N points)"
    )
    parser.add_argument(
        "--n-estimators", dest="n_estimators", type=int, default=300, help="Number of trees in RF"
    )
    parser.add_argument(
        "--max-depth", dest="max_depth", type=int, default=None, help="Max depth for RF"
    )
    args = parser.parse_args()
    return ForecastConfig(
        data_csv=args.data_csv,
        target_column=args.target_column,
        date_column=args.date_column,
        n_lags=args.n_lags,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_and_evaluate(cfg)
