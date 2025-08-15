# 🚗 Used Car Price Prediction (Rusty Bargain)

End‑to‑end workflow to build and compare regression models that estimate used car market value from historical listing data. Focus: accuracy (RMSE), training speed, and prediction latency.

## 📌 Project Overview
We analyze a vehicle listings dataset containing technical specs and categorical descriptors, clean it, explore distributions, engineer encodings, and benchmark multiple regressors to select a production‑ready baseline.

Notebook: `sprint_twelve_numerical_methods_nb.ipynb`

## 🗂️ Data Schema (after cleaning)
| Column | Type | Notes |
|--------|------|-------|
| price | target | Euro price (filtered: >500) |
| vehicle_type | categorical | Body style |
| registration_year | int | Filtered 1900–2025 |
| gearbox | categorical | manual / automatic / unknown |
| power | int | Filtered 40–400 hp |
| model | categorical | Many high‑cardinality values |
| kilometers | int | Odometer (renamed from mileage) |
| registration_month | int | 1–12 |
| fuel_type | categorical | petrol, diesel, others, unknown |
| brand | categorical | Manufacturer |
| not_repaired | categorical | yes / no / unknown |

Dropped: date crawl/creation/last_seen fields, number_of_pictures, postal_code.

## 🧹 Preprocessing Steps
1. Column normalization (lowercase + renames)
2. Drop non‑predictive / leakage columns
3. Fill categorical NaNs with 'unknown'
4. Remove duplicates
5. Filter implausible ranges:
   - price > 500
   - registration_year ∈ [1900, 2025]
   - power ∈ [40, 400]
6. Ordinal encode categorical features for tree & linear models
7. Train/validation/test split (64/16/20)

## 🔎 EDA Highlights
- Price: strong right skew; long upper tail.
- Power & price show plausible trimmed ranges after filtering.
- High cardinality in `model`; dominated by common brands (VW, Opel, BMW).
- `kilometers` exhibits stepped / rounded clusters (possible reporting granularity).

## 🧪 Models Benchmarked (RMSE, validation or noted split)
| Model | RMSE (~) | Fit Time | Predict Time |
|-------|----------|----------|--------------|
| Linear Regression | 2974 | ~0.02 s | ~0.01 s |
| Decision Tree (best depth 13) | 1957 | ~0.57 s | ~0.02 s |
| Random Forest (depth 17, 100 est) | 1649 | ~29.07 s | ~0.73 s |
| CatBoost (grid, depth 15) | 1972 | ~42.51 s (+grid 355 s) | ~0.02 s |
| LightGBM (tuned basic) | 1841 | ~2.28 s | ~0.19 s |
| XGBoost (validation) | 1626 | ~15.15 s | ~0.02 s |
| XGBoost (final test) | 1632 | ~14.36 s | ~0.02 s |

Metric: Root Mean Squared Error (lower better).

## 🏆 Selected Model
XGBoost Regressor chosen: best accuracy–speed balance and stable generalization (≈1.6k RMSE). Suitable for deployment with moderate latency and room for hyperparameter tuning (current run near defaults).

## 📦 Installation
```bash
git clone https://github.com/Vyncent-vdW/sprint_twelve_numerical_methods.git
cd sprint_twelve_numerical_methods
python -m venv .venv
./.venv/Scripts/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt  # If provided
# Or minimal set:
pip install pandas numpy scikit-learn seaborn matplotlib xgboost lightgbm catboost
```

## ▶️ Usage
```bash
jupyter notebook sprint_twelve_numerical_methods_nb.ipynb
```
Run sequentially:
1. Preprocessing & EDA
2. Encoding & splits
3. Model benchmarking
4. Final XGBoost evaluation

## 📐 Evaluation
Primary: RMSE on validation & test set.
Secondary (implied): training/prediction duration for operational feasibility.

## 🧩 Possible Enhancements
| Area | Idea |
|------|------|
| Feature Engineering | Age = current_year - registration_year; price per hp; interaction terms |
| Encoding | Target / frequency encoding for high‑cardinality `model` |
| Outliers | Robust trimming via IQR or quantile capping |
| Modeling | Hyperparameter search (XGBoost / LightGBM) |
| Validation | K-fold or time‑aware split if temporal leakage suspected |
| Interpretability | SHAP for feature impact |
| Deployment | Convert model to ONNX / pickle + inference script |
| Monitoring | Drift detection on categorical distributions & RMSE tracking |

## 📁 Repository Structure (core)
- sprint_twelve_numerical_methods_nb.ipynb – main workflow
- README.txt – project overview (this file)
- (Add requirements.txt / LICENSE as needed)

## 🔄 Reproducibility
- Deterministic preprocessing logic
- Fixed random_state where applicable (trees / splits)
- Explicit filtering rules documented above

## 📄 License
Add a LICENSE file (MIT / Apache-2.0 recommended).

## 🙏 Acknowledgments
Educational dataset context (used car listings) and open-source ML ecosystem: scikit-learn, XGBoost, LightGBM, CatBoost, pandas, seaborn.

## 🏁 Summary
A structured pipeline cleanses and analyzes listing data, then benchmarks classical and gradient boosting regressors. XGBoost currently offers the best RMSE with acceptable latency, forming a solid baseline for further tuning and explainability additions.