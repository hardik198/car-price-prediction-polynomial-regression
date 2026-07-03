# ==========================================
# Step 8 - Train Linear Regression Model
# ==========================================

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------------
# Project Path
# ------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "featured_data.csv"

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv(DATA_PATH)

# ------------------------------------------
# Features & Target
# ------------------------------------------

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# ------------------------------------------
# Train Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ------------------------------------------
# Train Model
# ------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

print("=" * 60)
print("LINEAR REGRESSION MODEL TRAINED")
print("=" * 60)

# ------------------------------------------
# Prediction
# ------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------
# Evaluation
# ------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"\nMAE  : {mae:.3f}")
print(f"MSE  : {mse:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# ------------------------------------------
# Compare Actual vs Predicted
# ------------------------------------------

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nFirst 10 Predictions")
print(results.head(10))