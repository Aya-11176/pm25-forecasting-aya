from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FOLDS_DIR = PROJECT_ROOT / "folds"

RAW_DATA_FILE = RAW_DIR / "combined_gee_pm25.csv"
CLEAN_DATA_FILE = PROCESSED_DIR / "PM25_Cleaned.csv"
FEATURE_DATA_FILE = PROCESSED_DIR / "PM25_Forecasting_Features_1day.csv"

SEED = 42
TEST_START = "2025-04-07"
TEST_END = "2026-04-06"

for directory in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, FOLDS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
