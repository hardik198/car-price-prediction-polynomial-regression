# ==========================================
# Step 13 - Machine Learning Pipeline
# ==========================================

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ------------------------------------------
# Paths
# ------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "featured_data.csv"

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# ------------------------------------------
# Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ------------------------------------------
# Pipeline
# ------------------------------------------

pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("model", LinearRegression())
])

# ------------------------------------------
# Train
# ------------------------------------------

pipeline.fit(X_train, y_train)

# ------------------------------------------
# Prediction
# ------------------------------------------

y_pred = pipeline.predict(X_test)

# ------------------------------------------
# Evaluation
# ------------------------------------------

print("=" * 60)
print("PIPELINE RESULTS")
print("=" * 60)

print(f"R² Score : {r2_score(y_test, y_pred):.4f}")
print(f"MAE      : {mean_absolute_error(y_test, y_pred):.4f}")