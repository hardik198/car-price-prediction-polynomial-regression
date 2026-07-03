# ==========================================
# Step 9 - Polynomial Regression
# ==========================================

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
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
# Polynomial Features
# ------------------------------------------

poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print("=" * 60)
print("POLYNOMIAL FEATURES")
print("=" * 60)

print("Original Features :", X_train.shape[1])
print("Polynomial Features :", X_train_poly.shape[1])

# ------------------------------------------
# Train Model
# ------------------------------------------

model = LinearRegression()

model.fit(X_train_poly, y_train)

# ------------------------------------------
# Prediction
# ------------------------------------------

y_pred = model.predict(X_test_poly)

# ------------------------------------------
# Evaluation
# ------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nResults")
print("-" * 40)

print(f"MAE  : {mae:.3f}")
print(f"MSE  : {mse:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# ------------------------------------------
# Compare Predictions
# ------------------------------------------

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nFirst 10 Predictions")

print(results.head(10))

print("\nPolynomial Feature Names")

feature_names = poly.get_feature_names_out(X.columns)

for feature in feature_names:
    print(feature)