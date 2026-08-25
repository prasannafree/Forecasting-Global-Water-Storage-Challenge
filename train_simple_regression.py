import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path("raw_data")

def compute_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }

def normalise_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" in df.columns and "ID" not in df.columns:
        return df.rename(columns={"sample_id": "ID"})
    return df

print("Loading data...")
train_features = normalise_id_column(pd.read_csv(DATA_DIR / "Train.csv", parse_dates=["time"]))
test_features = normalise_id_column(pd.read_csv(DATA_DIR / "Test.csv", parse_dates=["time"]))
sample_submission_df = pd.read_csv(DATA_DIR / "SampleSubmission.csv")

if 'target' in train_features.columns:
    train_features = train_features.rename(columns={'target': 'Target'})

available_common = set(train_features.columns).intersection(set(test_features.columns))

BASE_FEATURES = ["TWS_t", "month_sin", "month_cos"]
OPTIONAL_FEATURES = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]
feature_names = BASE_FEATURES + [f for f in OPTIONAL_FEATURES if f in available_common]

def build_xy(df: pd.DataFrame, feature_names: list):
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df["Target"].to_numpy(dtype=np.float32) if "Target" in df.columns else None
    return X, y

# Splitting for Validation
unique_times = np.sort(train_features["time"].unique())
split_idx = int(len(unique_times) * 0.8)

fit_times = unique_times[:split_idx]
val_times = unique_times[split_idx:]

fit_df = train_features[train_features["time"].isin(fit_times)]
val_df = train_features[train_features["time"].isin(val_times)]

X_fit, y_fit = build_xy(fit_df, feature_names)
X_val, y_val = build_xy(val_df, feature_names)

print("Training Simple Linear Regression model...")
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("lr", LinearRegression())
])

model.fit(X_fit, y_fit)

print("\n--- Validation Metrics ---")
y_val_pred = model.predict(X_val)
metrics = compute_metrics(y_val, y_val_pred)
print(f"MAE:  {metrics['mae']:.5f}")
print(f"RMSE: {metrics['rmse']:.5f}")
print(f"R2:   {metrics['r2']:.5f}")

# Train on full data for submission
print("\nTraining on full dataset for final submission...")
X_train_full, y_train_full = build_xy(train_features, feature_names)
final_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("lr", LinearRegression())
])
final_model.fit(X_train_full, y_train_full)

# Generate predictions
print("Generating test predictions...")
test_submission_df = sample_submission_df.merge(test_features, on="ID", how="left", validate="one_to_one")
X_test, _ = build_xy(test_submission_df, feature_names)

y_test_pred = final_model.predict(X_test)
submission_df = sample_submission_df.copy()
submission_df["Target"] = y_test_pred.astype(np.float32)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
out_path = results_dir / "SimpleRegression_Submission.csv"
submission_df.to_csv(out_path, index=False)
print(f"Submission saved to: {out_path}")
