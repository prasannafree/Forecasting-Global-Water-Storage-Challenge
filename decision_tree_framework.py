import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.impute import SimpleImputer

def main():
    print("Decision Tree Framework Implementation (Scikit-Learn)\n")
    
    DATA_DIR = Path("raw_data")
    
    def normalise_id_column(df: pd.DataFrame) -> pd.DataFrame:
        if "sample_id" in df.columns and "ID" not in df.columns:
            return df.rename(columns={"sample_id": "ID"})
        return df

    print("Loading dataset...")
    train_features = normalise_id_column(pd.read_csv(DATA_DIR / "Train.csv", parse_dates=["time"]))
    if 'target' in train_features.columns:
        train_features = train_features.rename(columns={'target': 'Target'})

    BASE_FEATURES = ["TWS_t", "month_sin", "month_cos"]
    OPTIONAL_FEATURES = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]
    available_features = BASE_FEATURES + [f for f in OPTIONAL_FEATURES if f in train_features.columns]

    X = train_features[available_features].to_numpy(dtype=np.float32)
    y = train_features["Target"].to_numpy(dtype=np.float32)

    unique_times = np.sort(train_features["time"].unique())
    split_idx = int(len(unique_times) * 0.8)

    fit_times = unique_times[:split_idx]
    val_times = unique_times[split_idx:]

    fit_mask = train_features["time"].isin(fit_times)
    val_mask = train_features["time"].isin(val_times)

    X_train, y_train = X[fit_mask], y[fit_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")

    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    print("Training DecisionTreeRegressor...")
    clf = DecisionTreeRegressor(max_depth=4, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print(f"\nFramework Decision Tree Validation Metrics:")
    print(f"MAE:  {mae:.5f}")
    print(f"RMSE: {rmse:.5f}")
    print(f"R2:   {r2:.5f}")

    print("\nGenerating visualization...")
    plt.figure(figsize=(15,10))
    tree.plot_tree(clf, feature_names=available_features, filled=True, rounded=True, precision=2)
    plt.savefig("framework_tree.png")
    print("Saved tree visualization to 'framework_tree.png'.")

if __name__ == "__main__":
    main()
