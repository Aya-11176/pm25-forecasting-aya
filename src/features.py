import numpy as np
import pandas as pd

PREDICTOR_COLUMNS = [
    "pm25", "AOD", "NO2", "CO", "O3", "temperature_celsius",
    "pressure_mb", "wind_u", "wind_v"
]

def build_one_day_ahead_features(dataframe):
    """
    Reproduce the notebook's past-only, one-day-ahead feature engineering.
    The target is PM2.5 on day t; predictors are shifted to information
    available no later than day t-1.
    """
    df = dataframe.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["id"] = df["id"].astype("string")
    predictor_cols = [c for c in PREDICTOR_COLUMNS if c in df.columns]

    feature_base = df.sort_values(["id", "date"]).copy()
    if feature_base.duplicated(["id", "date"]).any():
        examples = feature_base.loc[
            feature_base.duplicated(["id", "date"], keep=False), ["id", "date"]
        ].head(10)
        raise ValueError(
            "Duplicate station-date records exist. Resolve them before daily reindexing.\n"
            + examples.to_string(index=False)
        )

    feature_parts = []
    for station_id, group in feature_base.groupby("id", sort=False):
        group = group.sort_values("date").set_index("date")
        daily = group.reindex(pd.date_range(group.index.min(), group.index.max(), freq="D"))
        daily["id"] = station_id
        if "sensor_name" in group.columns and len(group):
            daily["sensor_name"] = group["sensor_name"].iloc[0]
        for column in ["latitude", "longitude"]:
            if column in group.columns:
                daily[column] = group[column].median()

        for column in predictor_cols:
            daily[f"{column}_lag1"] = daily[column].shift(1)

        if "pm25" in daily.columns:
            previous_pm25 = daily["pm25"].shift(1)
            daily["pm25_roll3_mean"] = previous_pm25.rolling(3, min_periods=3).mean()
            daily["pm25_roll3_std"] = previous_pm25.rolling(3, min_periods=3).std()
            daily["pm25_roll7_mean"] = previous_pm25.rolling(7, min_periods=5).mean()
            daily["pm25_roll7_max"] = previous_pm25.rolling(7, min_periods=5).max()

        for column in [c for c in ["AOD", "temperature_celsius", "NO2"] if c in daily.columns]:
            daily[f"{column}_diff1"] = daily[column].shift(1) - daily[column].shift(2)
            daily[f"{column}_diff3"] = daily[column].shift(1) - daily[column].shift(4)

        day_of_year = daily.index.dayofyear
        daily["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
        daily["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
        daily["month"] = daily.index.month
        daily["day_of_week"] = daily.index.dayofweek
        daily["date"] = daily.index

        daily = daily[daily["pm25"].notna()].copy()
        daily = daily[daily["pm25_lag1"].notna()].copy()
        feature_parts.append(daily.reset_index(drop=True))

    if not feature_parts:
        raise ValueError("No feature rows were produced. Check station timelines and PM2.5 data.")

    model_data = pd.concat(feature_parts, ignore_index=True)
    model_data = model_data.sort_values(["date", "id"]).reset_index(drop=True)
    return model_data


if __name__ == "__main__":
    from .config import CLEAN_DATA_FILE, FEATURE_DATA_FILE, RAW_DATA_FILE
    from .preprocessing import load_and_clean_csv

    if CLEAN_DATA_FILE.exists():
        cleaned = pd.read_csv(CLEAN_DATA_FILE, parse_dates=["date"])
    else:
        cleaned, _ = load_and_clean_csv(RAW_DATA_FILE, CLEAN_DATA_FILE)
    engineered = build_one_day_ahead_features(cleaned)
    engineered.to_csv(FEATURE_DATA_FILE, index=False)
    print(f"Saved {len(engineered):,} feature rows to: {FEATURE_DATA_FILE}")
