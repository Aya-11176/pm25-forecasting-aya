## pm25-forecasting-aya


This repo holds our EDA, time-series analysis and data preparation for a PM2.5 forecasting project. We forecast one day ahead: to predict PM2.5 on day *t*, we only use information available up to day *t-1*. A lot of the preparation work goes into making sure nothing from the future leaks into the features.

For now the pipeline is set up around Gradient Boosting. Model comparison, hyperparameter tuning, explainability and ablation come in later stages.

## Repository structure

```text
.
├── data/
│   ├── raw/                 # original Kaggle/GEE CSV (local only)
│   └── processed/           # cleaned and engineered datasets (local only)
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_timeseries.ipynb
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── features.py
│   └── splits.py
├── scripts/
│   └── run_eda.py
├── folds/
│   ├── folds_temporal.json
│   ├── folds_spatial.json
│   └── folds_standard.json
├── results/
│   ├── Exploratory_Data.csv
│   └── Exploratory_temporal.csv
├── figures/
├── reports/
└── presentation/
```

## Dataset

We start from `combined_gee_pm25.csv`, which has 41,718 rows and 15 columns. After cleaning, saved as `PM25_Cleaned.csv`, 40,983 rows remain. A row is kept if:

- PM2.5 is present, greater than 0 and below 500
- the date is valid
- the station ID is valid

The cleaned data runs from 2016-11-25 to 2026-04-06 and covers 95 stations.

**About the data and GitHub:** `data/raw/` and `data/processed/` are in `.gitignore`. The CSVs are in the project archive so you can run everything, but please don't push them to GitHub if the teacher wants the data to stay local. The `.gitkeep` files are there so the empty folders still show up in the repo.

## Setup

Python 3.10 or newer works best. Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Running the project

Run everything from the repository root, in this order.

1. Build the cleaned dataset and the EDA tables:

```bash
python scripts/run_eda.py
```

2. Build the one-day-ahead feature dataset:

```bash
python -m src.features
```

3. Generate the three sets of five folds (JSON):

```bash
python -m src.splits
```

The fold indices point to rows of the development dataset after it has been sorted and its index reset. The final test period is held out entirely and covers 2025-04-07 to 2026-04-06. The random KFold file is only there as a diagnostic. For the actual forecasting evaluation, use the temporal and spatial folds.

## Notebooks

- `01_eda_v2.ipynb`: exploratory analysis, covering summaries, missing values, distributions, outliers and correlations.
- `02_timeseries_v2.ipynb`: temporal diagnostics, feature engineering that only looks at the past, leakage checks and the validation protocol.

Both were split out of our original combined notebook, and the `_v2` files replace the first versions (`01_eda` and `02_timeseries`) with improved ones.

They run on Google Colab. When prompted, upload the raw `combined_gee_pm25.csv`.

## Outputs

- `results/Exploratory_Data.csv`: descriptive statistics
- `results/Exploratory_temporal.csv`: temporal diagnostics per station
- `folds/*.json`: the five train/validation splits for temporal, spatial and random (diagnostic) cross-validation
