# Data Report — Hybrid AI Fraud Detection System (Task 1)
## Dataset
- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Filename: `creditcard.csv`
- Rows (after duplicate removal): 283,726
- Columns: 31

## Target
- Target column: `Class` (0 = legitimate, 1 = fraud)
- Class 0 count: 283,253 (99.8333%)
- Class 1 count: 473 (0.1667%)
- Imbalance ratio (legit:fraud): 598.8 : 1

## Data Quality
- Missing values: None found
- Duplicate rows found: 1,081 (0.3796%) - dropped before splitting

## Leakage Prevention
- Target dropped from X immediately after split; never used as an input feature.
- Train/validation/test split performed BEFORE fitting the scaler.
- Scaler fit only on the training split.
- Duplicate rows removed before splitting, preventing duplicate leakage across splits.
- Engineered features (`Hour_sin`, `Hour_cos`, `Amount_log`) computed row-wise only, no cross-row or label information used.

## Split
- Train: 198,608 rows (70.00%), fraud rate 0.1667%
- Validation: 42,559 rows (15.00%), fraud rate 0.1668%
- Test: 42,559 rows (15.00%), fraud rate 0.1668%
- Stratified on `Class`, random_state=42

## Preprocessing
- Scaled columns (StandardScaler, fit on train only): ['Time', 'Amount', 'Amount_log']
- Passthrough columns (already PCA'd by dataset authors): V1-V28
- Saved artifact: `artifacts/preprocessing_pipeline.pkl`

## Features
- Final feature count: 33
- Final feature order: ['Time', 'Amount', 'Amount_log', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Hour_cos', 'Hour_sin']
- Engineered features: `Hour_sin`, `Hour_cos` (cyclical hour-of-day from `Time`), `Amount_log` (log1p of `Amount`) - all inference-safe.

## Handoff
**Person 2 (supervised ML)** loads `data/processed/{train,validation,test}.csv`. Each file's columns are the fixed feature order from `artifacts/feature_config.json` followed by `Class`. Load `artifacts/preprocessing_pipeline.pkl` (via `FraudPreprocessor.load`) to transform any new raw transaction identically at inference time.

**Person 3 (Autoencoder)** trains on the legitimate-only subset: `X_train_normal = train_df[train_df.Class == 0].drop(columns=['Class'])`, using the same feature representation.
