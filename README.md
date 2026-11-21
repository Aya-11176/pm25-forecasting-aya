pm25-forecasting-aya
This repository contains the EDA, time-series analysis and data-preparation workflow for a PM2.5 forecasting project. The task is one-day-ahead forecasting: to predict PM2.5 on day t, model inputs must be available no later than day t-1.

The current pipeline prepares the data and validation folds for the assigned Gradient Boosting model. Model comparison, hyper-parameter tuning, explainability and ablation are later project stages.
Repository structure
.
├── data/                 # raw/ and processed/ data are local; CSVs are not included
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_timeseries.ipynb
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── features.py
│   └── splits.py
├── scripts/run_eda.py
├── folds/                # saved temporal, spatial and diagnostic random folds
├── results/              # committed summary tables
├── figures/              # committed EDA figures
├── reports/
└── presentation/
Dataset
The expected input file is combined_gee_pm25.csv, placed at data/raw/combined_gee_pm25.csv for the Python scripts, or uploaded when running either notebook in Google Colab. The raw CSV is not included in this repository archive. Obtain it from the approved project source and verify its license and variable units before redistribution.

The project cleaning rule retains rows with PM2.5 present and strictly between 0 and 500, plus valid dates and station IDs. The README's previously reported row/station counts are dataset-dependent and should be confirmed by rerunning the pipeline against the exact source CSV.
Setup
Python 3.10 or newer is recommended.
python -m pip install -r requirements
Running the project
Run commands from the repository root after placing the source CSV at data/raw/combined_gee_pm25.csv.
1. Clean the data and build EDA summary tables:
python scripts/run_eda.py
2. Build the one-day-ahead feature dataset:
python -m src.features
3. Generate the temporal, spatial and random-diagnostic folds:
python -m src.splits
src.features.get_model_feature_columns() returns the explicit whitelist of past-only predictors. Raw same-day measurements and station metadata may remain in the engineered table for auditing, but must not be passed to the model. The expected count is 23 when all nine source predictors and all engineered feature families are available; the actual count is printed at runtime.

Fold indices refer to rows of the development dataset after sorting and resetting the index. The final future test period is 2025-04-07 through 2026-04-06. Random K-fold is diagnostic only; temporal and spatial folds are the primary validation schemes.
Notebooks
- notebooks/01_eda.ipynb: cleaning checks, descriptive analysis, outlier diagnostics, correlations and exploratory figures. Lag 0 in the lead–lag plot is descriptive only and is not a forecasting input.
- notebooks/02_timeseries.ipynb: temporal diagnostics, past-only feature engineering, explicit leakage-safe feature selection and validation protocol.
Both notebooks can be run in Google Colab by uploading the raw CSV when prompted.
