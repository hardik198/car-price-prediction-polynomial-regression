# ==========================================
# Step 11 - Save Best Polynomial Model
# ==========================================

import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# ------------------------------------------
# Project Paths
# ------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "featured_data.csv"

MODEL_FOLDER = PROJECT_ROOT / "models"

MODEL_FOLDER.mkdir(exist_ok=True)

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop("Selling_Price", axis=1)

y = df["Selling_Price"]

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
    degree=3,
    include_bias=False
)

X_train_poly = poly.fit_transform(X_train)

# ------------------------------------------
# Train Model
# ------------------------------------------

model = LinearRegression()

model.fit(X_train_poly, y_train)

# ------------------------------------------
# Save Model
# ------------------------------------------

joblib.dump(
    model,
    MODEL_FOLDER / "best_model.pkl"
)

joblib.dump(
    poly,
    MODEL_FOLDER / "polynomial_features.pkl"
)

print("Model Saved Successfully")