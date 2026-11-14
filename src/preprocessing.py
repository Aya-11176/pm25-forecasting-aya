import pandas as pd

ALIASES = {
    "PM2.5": "pm25", "PM25": "pm25", "PM_25": "pm25",
    "station_id": "id", "station": "id", "station_name": "sensor_name",
    "lat": "latitude", "lon": "longitude", "lng": "longitude",
    "temperature": "temperature_celsius", "temp": "temperature_celsius",
    "pressure": "pressure_mb", "wind_u_component": "wind_u", "wind_v_component": "wind_v",
    "no2": "NO2", "co": "CO", "o3": "O3"
}

NUMERIC_CANDIDATES = [
    "latitude", "longitude", "temperature_celsius", "pressure_mb",
    "wind_u", "wind_v", "NO2", "CO", "O3", "AOD", "pm25"
]

def standardize_columns(dataframe):
    """Standardize known column aliases and normalize data types."""
    df = dataframe.rename(
        columns={column: ALIASES[column] for column in dataframe.columns if column in ALIASES}
    ).copy()

    required = ["date", "id", "pm25"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if "datetime_utc" in df.columns:
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")

    for column in NUMERIC_CANDIDATES:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["id"] = df["id"].astype("string")
    if "sensor_name" in df.columns:
        df["sensor_name"] = df["sensor_name"].astype("string")
    return df

def clean_pm25_dataframe(dataframe):
    """
    Apply the project's documented mandatory cleaning:
    keep rows with PM2.5 > 0 and < 500, valid dates and station IDs.
    """
    df = standardize_columns(dataframe)
    initial_rows = len(df)
    counts = {
        "initial_rows": initial_rows,
        "missing_pm25": int(df["pm25"].isna().sum()),
        "pm25_le_zero": int((df["pm25"].notna() & df["pm25"].le(0)).sum()),
        "pm25_ge_500": int((df["pm25"].notna() & df["pm25"].ge(500)).sum()),
        "invalid_date": int(df["date"].isna().sum()),
        "missing_station_id": int(df["id"].isna().sum()),
    }

    df = df[df["pm25"].notna() & df["pm25"].gt(0) & df["pm25"].lt(500)].copy()
    df = df[df["date"].notna() & df["id"].notna()].copy()
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    counts["rows_retained"] = len(df)
    return df, counts

def load_and_clean_csv(input_path, output_path=None):
    raw = pd.read_csv(input_path)
    cleaned, counts = clean_pm25_dataframe(raw)
    if output_path is not None:
        cleaned.to_csv(output_path, index=False)
    return cleaned, counts
