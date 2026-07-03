# ==========================================
# Step 10 - Compare Polynomial Degrees
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
# Project Paths
# ------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "featured_data.csv"

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv(DATA_PATH)

# ------------------------------------------
# Features and Target
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
# Results List
# ------------------------------------------

results = []

# ------------------------------------------
# Degree Loop
# ------------------------------------------

for degree in [1, 2, 3, 4, 5]:

    if degree == 1:

        X_train_new = X_train

        X_test_new = X_test

    else:

        poly = PolynomialFeatures(
            degree=degree,
            include_bias=False
        )

        X_train_new = poly.fit_transform(X_train)

        X_test_new = poly.transform(X_test)

    model = LinearRegression()

    model.fit(X_train_new, y_train)

    y_pred = model.predict(X_test_new)

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Degree": degree,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2 Score": round(r2, 3)
    })

# ------------------------------------------
# Result Table
# ------------------------------------------

results_df = pd.DataFrame(results)

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df)

best = results_df.loc[
    results_df["R2 Score"].idxmax()
]

print("\nBest Model")

print(best)


import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(
    results_df["Degree"],
    results_df["R2 Score"],
    marker="o"
)

plt.title("Polynomial Degree vs R2 Score")

plt.xlabel("Degree")

plt.ylabel("R2 Score")

plt.grid(True)

plt.savefig(
    PROJECT_ROOT / "images" / "degree_vs_r2.png"
)

plt.show()