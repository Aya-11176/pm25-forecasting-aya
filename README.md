
# PM2.5 Forecasting — EDA & Time-Series Analysis

This repository contains the Exploratory Data Analysis (EDA), time-series analysis, feature engineering, and data-preparation workflow for a PM2.5 forecasting project.

The task is **one-day-ahead forecasting**: to predict PM2.5 on day *t*, all model inputs must be available no later than day *t − 1*.

The current pipeline prepares the dataset, constructs past-only predictors, and generates validation folds for the assigned Gradient Boosting model. Model comparison, hyperparameter tuning, explainability, and ablation studies belong to later project stages.

---

## Repository Structure

```text
pm25-forecasting-aya/
│
├── data/
│   ├── raw/                 # Original dataset (not included)
│   └── processed/           # Generated processed datasets
│
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   └── 02_timeseries.ipynb  # Time-Series Analysis & Validation
│
├── src/
│   ├── config.py            # Project configuration
│   ├── preprocessing.py     # Data cleaning and preprocessing
│   ├── features.py          # Feature engineering and selection
│   └── splits.py            # Validation fold generation
│
├── scripts/
│   └── run_eda.py           # EDA pipeline entry point
│
├── folds/                   # Saved validation folds
├── results/                 # Generated summary tables
├── figures/                 # EDA figures
├── reports/                 # Project reports
├── presentation/            # Project presentation
│
└── README.md
```

---

## Dataset

The project expects the following input file:

`combined_gee_pm25.csv`

For local execution, place it at:

```text
data/raw/combined_gee_pm25.csv
```

When using Google Colab, upload the original CSV when required by the notebook.

**Note:** The raw dataset is not included in this repository. Obtain it from the approved project source and verify its license and variable units before redistribution.

### Data Cleaning

The preprocessing workflow retains observations satisfying the following conditions:

- PM2.5 is present.
- PM2.5 is strictly between 0 and 500.
- The date is valid.
- The station ID is available.

Dataset statistics, including the number of retained observations and stations, should be confirmed by rerunning the pipeline using the exact source CSV.

---

## Installation & Setup

**Requirements:** Python 3.10 or newer.

Install the project dependencies from the repository root:

```bash
python -m pip install -r requirements
```

---

## Running the Project

Before running the scripts, make sure the source CSV is available at:

```text
data/raw/combined_gee_pm25.csv
```

Run the following commands from the repository root, in order.

### 1. Data Preprocessing & EDA

```bash
python scripts/run_eda.py
```

This step performs data cleaning and generates exploratory summary tables and figures.

### 2. Feature Engineering

```bash
python -m src.features
```

This step constructs the one-day-ahead feature dataset using historical observations.

### 3. Validation Fold Generation

```bash
python -m src.splits
```

This step generates the temporal, spatial, and random diagnostic validation folds.

---

## Feature Engineering & Leakage Prevention

The forecasting setup follows a past-only information rule:

**To predict PM2.5 on day *t*, the model must not use measurements from day *t*.**

The feature-selection function:

`src.features.get_model_feature_columns()`

uses an explicit whitelist of eligible predictors.

The selected features include:

- Historical lagged measurements.
- Past-only rolling statistics.
- Historical differences.
- Calendar variables known in advance.

The following variables are excluded from model inputs:

- Raw same-day environmental measurements.
- Target-day PM2.5.
- Station identifiers.
- Latitude and longitude.
- Isolation Forest anomaly flags.

The expected number of eligible features is **23**, provided that all nine source predictors and the required engineered feature columns are available. The actual number is printed when the pipeline runs.

Raw measurements and metadata may remain in the engineered dataset for inspection and auditing, but they must not be passed to the forecasting model.

---

## Validation Strategy

The project uses multiple validation approaches:

- **Temporal validation:** evaluates generalization across time.
- **Spatial validation:** evaluates generalization across stations.
- **Random K-Fold:** used for diagnostic comparison only.

Temporal and spatial validation are the primary evaluation strategies.

Fold indices refer to rows of the development dataset after sorting and resetting the index.

The final future test period is:

`2025-04-07` to `2026-04-06`.

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | Data cleaning checks, descriptive statistics, outlier diagnostics, correlations, and exploratory figures. |
| `02_timeseries.ipynb` | Temporal diagnostics, past-only feature engineering, leakage-safe feature selection, and validation protocol. |

**Important:** Lag 0 in the exploratory lead–lag analysis is descriptive only. It is not used as a forecasting input.

Both notebooks can be executed in Google Colab by uploading the original CSV when required.

---

## Project Status

The current stage focuses on:

- Data preprocessing.
- Exploratory Data Analysis.
- Time-series analysis.
- Past-only feature engineering.
- Leakage prevention.
- Temporal and spatial validation preparation.

Model fitting, hyperparameter optimization, model comparison, explainability, and ablation studies are subsequent stages of the project.
