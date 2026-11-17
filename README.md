# State-wise Secondary Dropout Analysis (UDISE+ 2023-24)

## Overview
This project analyzes the determinants of secondary school dropout rates across Indian states and UTs using UDISE+ 2023-24 indicators. It implements a complete regression pipeline: data merging, correlation analysis, multicollinearity diagnostics (VIF), OLS regression with outlier detection, regularized regression (Ridge/Lasso), polynomial features, and leave-one-out cross-validation (LOOCV) model comparison.

## Data Source
- **Primary dataset**: `merged_dataset.csv` (36 states/UTs, 5 predictors + target)
- **Source**: UDISE+ 2023-24 (Unified District Information System for Education Plus)
- **Target**: `dropout_sec` — Secondary Drop Out Rate Overall (%)
- **Features**:
  - `PTR_sec` — Pupil Teacher Ratio (Secondary, grades 9-10)
  - `GER_sec` — Gross Enrolment Ratio Secondary (grades 9-10) 2023-24 (%)
  - `pct_electricity` — % Schools with Electricity Connection
  - `pct_computer` — % Schools with Computer Facility
  - `pct_internet` — % Schools with Internet Facility

## Methodology

### 1. Data Preparation
- State name normalization for consistent merging across sources
- Exclusion of national-level aggregates ("India", "National Average")
- Missing value removal (complete case analysis)

### 2. Exploratory Analysis
- Descriptive statistics (count, mean, std, min, quartiles, max)
- Pearson correlation matrix with heatmap visualization

### 3. Multicollinearity Diagnostics
- Variance Inflation Factor (VIF) for all predictors
- VIF < 5 indicates acceptable multicollinearity

### 4. Ordinary Least Squares (OLS) Regression
- Full model with all 36 states/UTs
- Cook's distance and studentized residuals for outlier detection
- Sensitivity analysis excluding 6 small UTs (Lakshadweep, Ladakh, Andaman & Nicobar, Chandigarh, Puducherry, Dadra & Nagar Haveli and Daman & Diu)

### 5. Regularized Regression
- **RidgeCV**: L2 regularization with cross-validated alpha selection
- **LassoCV**: L1 regularization with cross-validated alpha selection
- Standardized coefficients for effect size comparison

### 6. Polynomial Features
- Degree-2 polynomial expansion of top 3 correlated predictors
- Ridge regression on polynomial features

### 7. Model Comparison via LOOCV
Leave-one-out cross-validation comparing:
- Mean baseline
- OLS
- Ridge
- Lasso
- Poly-2 + Ridge (top 3 features)

Metrics: MAE, RMSE, R²

## Key Findings

### Correlation with Dropout Rate
| Predictor | Correlation |
|-----------|-------------|
| pct_internet | -0.479 |
| pct_computer | -0.434 |
| pct_electricity | -0.415 |
| GER_sec | -0.381 |
| PTR_sec | +0.317 |

Internet access shows the strongest negative correlation with dropout rates.

### Multicollinearity (VIF)
| Feature | VIF |
|---------|-----|
| PTR_sec | 1.28 |
| GER_sec | 1.55 |
| pct_electricity | 1.89 |
| pct_computer | 3.30 |
| pct_internet | 3.29 |

All VIFs < 5, indicating no severe multicollinearity. Infrastructure variables (computer, internet, electricity) show moderate correlation (0.6-0.8).

### OLS Results (All States, n=36)
- **R² = 0.362**, Adj. R² = 0.255, F-test p = 0.015
- Only **PTR_sec** shows marginal significance (p = 0.067)
- Infrastructure variables not individually significant in multivariate model

### Outlier Detection
Two influential states flagged (Cook's D > 4/n = 0.111):
1. **Dadra & Nagar Haveli and Daman & Diu** (Cook's D = 0.114)
2. **Uttar Pradesh** (Cook's D = 0.132)

### LOOCV Model Comparison (All States)
| Model | MAE | RMSE | LOOCV R² |
|-------|-----|------|----------|
| Ridge | **4.00** | **4.77** | **0.166** |
| OLS | 4.07 | 4.89 | 0.126 |
| Poly-2 + Ridge | 4.36 | 5.12 | 0.040 |
| Lasso | 4.35 | 5.19 | 0.014 |
| Mean baseline | 4.55 | 5.38 | -0.058 |

**Ridge regression performs best** in leave-one-out prediction.

### Standardized Coefficients (Ridge vs Lasso)
| Feature | Ridge | Lasso |
|---------|-------|-------|
| PTR_sec | +0.949 | +1.494 |
| GER_sec | -0.532 | -0.261 |
| pct_electricity | -0.808 | -1.224 |
| pct_computer | -0.506 | 0.000 |
| pct_internet | -0.873 | -1.468 |

Lasso shrinks `pct_computer` to zero, selecting PTR_sec, pct_internet, and pct_electricity as most important.

### Sensitivity Analysis (Excluding Small UTs, n=30)
- R² remains similar (0.361 vs 0.362)
- Ridge still outperforms (LOOCV R² = 0.134)

## Output Files

All outputs saved to `results/`:

| File | Description |
|------|-------------|
| `descriptives.csv` | Summary statistics for all variables |
| `correlation_matrix.csv` | Pearson correlation matrix |
| `fig_correlation.png` | Correlation heatmap |
| `vif.csv` | Variance Inflation Factors |
| `ols_all.csv` | OLS coefficients (all states) |
| `ols_no_small_UTs.csv` | OLS coefficients (excl. small UTs) |
| `fig_cooks.png` | Cook's distance bar plot |
| `outliers_flagged.csv` | Influential states table |
| `loocv_all.csv` | LOOCV metrics (all states) |
| `loocv_no_small_UTs.csv` | LOOCV metrics (excl. small UTs) |
| `standardised_coefficients.csv` | Ridge/Lasso standardized coefficients |
| `fig_coefficients.png` | Standardized coefficient comparison plot |
| `fig_obs_vs_pred.png` | Observed vs predicted (LOOCV Ridge) |

## Requirements
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
scipy>=1.10.0
patsy>=0.5.6
formulaic>=1.1.0
```

Install with:
```bash
pip install -r requirements.txt
```

## Usage

```bash
# Ensure merged_dataset.csv is in the project directory
python dropout_analysis.py
```

Outputs are written to `results/`.

## Project Structure
```
ML project/
├── dropout_analysis.py      # Main analysis script
├── merged_dataset.csv       # Input data (36 states/UTs × 6 variables)
├── ml_dropout_analysis.pdf  # Reference documentation
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── results/                 # Generated outputs (created on run)
    ├── *.csv                # Tables
    └── *.png                # Figures
```

## Interpretation Notes

1. **Causality**: This is observational cross-sectional analysis; correlations do not imply causation.

2. **Sample size**: n=36 (states/UTs) limits statistical power and model complexity.

3. **Multicollinearity**: Infrastructure variables are correlated; Ridge handles this better than OLS.

4. **Small UTs**: The 6 small UTs have extreme values on some indicators; sensitivity analysis shows robust qualitative conclusions.

5. **Best predictive model**: Ridge regression with all 5 predictors achieves the highest LOOCV R² (0.166), though predictive performance remains modest.

## License
Educational/research use. Data sourced from UDISE+ (Government of India).