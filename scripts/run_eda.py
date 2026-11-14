from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.config import RAW_DATA_FILE, CLEAN_DATA_FILE, RESULTS_DIR
from src.preprocessing import load_and_clean_csv

def create_eda_tables(df):
    numeric_cols = [
        c for c in [
            "pm25", "AOD", "NO2", "CO", "O3", "temperature_celsius",
            "pressure_mb", "wind_u", "wind_v", "latitude", "longitude"
        ] if c in df.columns
    ]
    summary = df[numeric_cols].describe(percentiles=[.25, .5, .75]).T
    summary = summary.rename(columns={"25%": "q25", "50%": "median", "75%": "q75"})
    summary["skewness"] = df[numeric_cols].skew()
    summary["kurtosis_excess"] = df[numeric_cols].kurtosis()
    summary.to_csv(RESULTS_DIR / "Exploratory_Data.csv", encoding="utf-8")

    temporal_rows = []
    for station_id, group in df.groupby("id"):
        group = group.sort_values("date").drop_duplicates("date", keep="first")
        series = group.set_index("date")["pm25"].asfreq("D")
        observed = series.dropna()
        if observed.empty:
            continue
        intervals = observed.index.to_series().diff().dt.total_seconds().dropna()
        full_days = len(series)
        missing_days = int(series.isna().sum())

        mask = series.notna().to_numpy()
        best_start = best_len = run_start = run_len = 0
        for i, is_observed in enumerate(mask):
            if is_observed:
                if run_len == 0:
                    run_start = i
                run_len += 1
                if run_len > best_len:
                    best_start, best_len = run_start, run_len
            else:
                run_len = 0
        segment = series.iloc[best_start:best_start + best_len].dropna().to_numpy(float)

        acf1 = acf7 = pacf1 = lb_p = adf_p = np.nan
        if len(observed) >= 3:
            acf1 = series.autocorr(1)
        if len(observed) >= 8:
            acf7 = series.autocorr(7)
        if len(segment) >= 30 and np.nanstd(segment) > 0:
            try:
                pacf1 = float(pacf(segment, nlags=1, method="ywm")[1])
            except Exception:
                pass
            try:
                lb_lag = min(7, max(1, len(segment) // 5))
                lb_p = float(acorr_ljungbox(segment, lags=[lb_lag], return_df=True)["lb_pvalue"].iloc[0])
            except Exception:
                pass
            try:
                adf_p = float(adfuller(segment, autolag="AIC")[1])
            except Exception:
                pass

        temporal_rows.append({
            "id": str(station_id),
            "n_observed": len(observed),
            "first_date": observed.index.min(),
            "last_date": observed.index.max(),
            "expected_daily_dates": full_days,
            "missing_calendar_days": missing_days,
            "missing_calendar_days_pct": missing_days / full_days * 100 if full_days else np.nan,
            "median_observed_interval_days": intervals.median() / 86400 if len(intervals) else np.nan,
            "duplicate_timestamps_before_station_dedup": int(group.duplicated("date").sum()),
            "longest_uninterrupted_segment_days": len(segment),
            "acf_lag1": acf1, "acf_lag7": acf7, "pacf_lag1": pacf1,
            "ljung_box_p": lb_p, "adf_p": adf_p,
            "pct_above_25": (observed > 25).mean() * 100,
            "pct_above_50": (observed > 50).mean() * 100
        })
    temporal = pd.DataFrame(temporal_rows)
    temporal.to_csv(RESULTS_DIR / "Exploratory_temporal.csv", index=False)
    return summary, temporal

def main():
    cleaned, cleaning_counts = load_and_clean_csv(RAW_DATA_FILE, CLEAN_DATA_FILE)
    summary, temporal = create_eda_tables(cleaned)
    print(f"Cleaned data saved to: {CLEAN_DATA_FILE}")
    print(f"Rows retained: {len(cleaned):,}")
    print(f"EDA tables saved to: {RESULTS_DIR}")
    print(f"Stations with ACF(1): {temporal.acf_lag1.notna().sum()}")
    print(f"Stations with ADF result: {temporal.adf_p.notna().sum()}")

if __name__ == "__main__":
    main()
