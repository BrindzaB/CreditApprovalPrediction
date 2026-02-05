# Credit Approval Prediction (Kaggle)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-6A0DAD?style=flat-square)](https://imbalanced-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-8B0000?style=flat-square)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF2D55?style=flat-square)](https://shap.readthedocs.io/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557C?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12%2B-4C72B0?style=flat-square)](https://seaborn.pydata.org/)

## Project Snapshot
- **Goal**: Classify customers as "good" vs "bad" credit risk.
- **Data**: Kaggle-style `application_record.csv` + `credit_record.csv` merged by `ID`.
- **Models**: Logistic Regression, Random Forest, XGBoost, k‑NN.
- **Explainability**: SHAP summary plot for global feature impact.

## What’s Inside
1. **Data Engineering**
- Aggregates monthly credit history into a single `is_bad` label per customer.
- Merges application + credit history into a unified dataset.

2. **Cleaning & Preprocessing**
- Fixes `DAYS_EMPLOYED` and `DAYS_BIRTH` anomalies.
- Handles missing `OCCUPATION_TYPE` with an "Unknown" category.
- Encodes categorical features with One‑Hot Encoding.
- Scales numerical features.

3. **Modeling**
- Splits data into train/test with stratification.
- Applies SMOTE to balance the training set only.
- Trains and evaluates 4 baseline models.

4. **Evaluation**
- Reports `accuracy`, `precision`, `recall`, and `F1`.
- Discusses tradeoffs under class imbalance.

5. **Explainability (SHAP)**
- Computes SHAP values for the tree model.
- Uses `summary_plot` to visualize global feature importance and direction.

## Key Results (Example)
Performance varies by metric due to class imbalance. Tree‑based models generally achieve high accuracy, while recall/precision tradeoffs differ:
- **Logistic Regression**: higher recall, low precision.
- **Random Forest / XGBoost**: stronger precision, weaker recall.
- **k‑NN**: improved recall vs trees, but lower precision.

## How to Run
1. Create a virtual environment and install dependencies.
2. Place data files in `data/`:
- `application_record.csv`
- `credit_record.csv`
3. Open and run the notebook in order.

## Files
- `Credit_Card_Approval_Prediction.ipynb`
- `Credit_Card_Approval_Prediction_EN.ipynb`
- `data/application_record.csv`
- `data/credit_record.csv`

## Notes
- The dataset is highly imbalanced, so accuracy alone is not sufficient.
- SHAP plots require numeric feature matrices; ensure the transformed DataFrame is numeric.
