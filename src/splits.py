import json
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, KFold

from .config import (
    CLEAN_DATA_FILE, FEATURE_DATA_FILE, FOLDS_DIR, TEST_START, TEST_END, SEED
)
from .features import build_one_day_ahead_features
from .preprocessing import load_and_clean_csv
from .config import RAW_DATA_FILE

def _json_ready(indices):
    return [int(value) for value in indices]

def create_and_save_folds(model_data, output_dir=FOLDS_DIR):
    """
    Create five temporal, spatial and standard random diagnostic folds.
    Indices refer to rows in the development dataset only (after sorting).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data = model_data.copy().reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"])
    data["id"] = data["id"].astype("string")

    test_start = pd.Timestamp(TEST_START)
    test_end = pd.Timestamp(TEST_END)
    development = data[data.date < test_start].copy().reset_index(drop=True)
    final_test = data[data.date.between(test_start, test_end)].copy().reset_index(drop=True)

    if development.empty or final_test.empty:
        raise ValueError(
            "Development or final test set is empty. "
            "Check TEST_START/TEST_END against the dataset coverage."
        )

    common_metadata = {
        "n_splits": 5,
        "index_base": 0,
        "indices_refer_to": "development_data rows after reset_index(drop=True)",
        "development_rows": int(len(development)),
        "development_date_min": str(development.date.min().date()),
        "development_date_max": str(development.date.max().date()),
        "final_test_start": str(test_start.date()),
        "final_test_end": str(test_end.date()),
        "final_test_rows": int(len(final_test)),
        "seed": int(SEED)
    }
    calendar = pd.date_range(development.date.min(), development.date.max(), freq="D")
    splitter = TimeSeriesSplit(n_splits=5, gap=1)
    temporal_folds = {}
    for fold, (train_dates_idx, valid_dates_idx) in enumerate(splitter.split(calendar), start=1):
        train_dates = calendar[train_dates_idx]
        valid_dates = calendar[valid_dates_idx]
        train_idx = np.flatnonzero(development.date.isin(train_dates).to_numpy())
        valid_idx = np.flatnonzero(development.date.isin(valid_dates).to_numpy())
        temporal_folds[str(fold)] = {
            "train_indices": _json_ready(train_idx),
            "validation_indices": _json_ready(valid_idx),
            "train_start": str(development.iloc[train_idx].date.min().date()),
            "train_end": str(development.iloc[train_idx].date.max().date()),
            "validation_start": str(development.iloc[valid_idx].date.min().date()),
            "validation_end": str(development.iloc[valid_idx].date.max().date()),
            "calendar_gap_days": int((valid_dates.min() - train_dates.max()).days - 1)
        }

    spatial_folds = {}
    gkf = GroupKFold(n_splits=5)
    placeholder = np.zeros((len(development), 1))
    groups = development.id.to_numpy()
    for fold, (train_idx, valid_idx) in enumerate(gkf.split(placeholder, groups=groups), start=1):
        train_groups = set(groups[train_idx])
        valid_groups = set(groups[valid_idx])
        if train_groups.intersection(valid_groups):
            raise AssertionError("A station appears in both train and validation groups.")
        spatial_folds[str(fold)] = {
            "train_indices": _json_ready(train_idx),
            "validation_indices": _json_ready(valid_idx),
            "train_stations": len(train_groups),
            "validation_stations": len(valid_groups),
            "shared_stations": 0
        }
    standard_folds = {}
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(development), start=1):
        standard_folds[str(fold)] = {
            "train_indices": _json_ready(train_idx),
            "validation_indices": _json_ready(valid_idx)
        }

    payloads = [
        ("folds_temporal.json", "Temporal TimeSeriesSplit with one-day gap", temporal_folds),
        ("folds_spatial.json", "Spatial GroupKFold grouped by station ID", spatial_folds),
        ("folds_standard.json", "Random KFold; diagnostic only, not primary evaluation", standard_folds)
    ]
    for filename, description, folds in payloads:
        payload = {
            "description": description,
            "metadata": common_metadata,
            "folds": folds
        }
        with open(output_dir / filename, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    return development, final_test

def main():
    if FEATURE_DATA_FILE.exists():
        model_data = pd.read_csv(FEATURE_DATA_FILE, parse_dates=["date"])
    else:
        if CLEAN_DATA_FILE.exists():
            cleaned = pd.read_csv(CLEAN_DATA_FILE)
            cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
        else:
            cleaned, _ = load_and_clean_csv(RAW_DATA_FILE, CLEAN_DATA_FILE)
        model_data = build_one_day_ahead_features(cleaned)
        model_data.to_csv(FEATURE_DATA_FILE, index=False)
    create_and_save_folds(model_data)
    print(f"Saved temporal, spatial and standard folds to: {FOLDS_DIR}")

if __name__ == "__main__":
    main()
