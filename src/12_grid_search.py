# ==========================================
# Step 14 - Hyperparameter Tuning
# ==========================================

import pandas as pd
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

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
    ("poly", PolynomialFeatures(include_bias=False)),
    ("model", LinearRegression())
])

# ------------------------------------------
# Parameter Grid
# ------------------------------------------

parameters = {
    "poly__degree": [1, 2, 3, 4, 5]
}

# ------------------------------------------
# Grid Search
# ------------------------------------------

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=parameters,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("=" * 60)
print("GRID SEARCH RESULTS")
print("=" * 60)

print()

print("Best Degree")

print(grid.best_params_)

print()

print("Best Cross Validation Score")

print(round(grid.best_score_,4))

print()

print("Test Score")

print(round(grid.score(X_test,y_test),4))