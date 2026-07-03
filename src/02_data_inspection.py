# ==========================================
# Step 3 - Data Inspection
# ==========================================

import pandas as pd
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "car data.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATA INSPECTION")
print("=" * 60)

# -----------------------------
# Missing Values
# -----------------------------
print("\n1. Missing Values")
print(df.isnull().sum())

# -----------------------------
# Duplicate Rows
# -----------------------------
print("\n2. Duplicate Rows")
print(df.duplicated().sum())

# -----------------------------
# Unique Values
# -----------------------------
print("\n3. Unique Values")
print(df.nunique())

# -----------------------------
# Data Types
# -----------------------------
print("\n4. Data Types")
print(df.dtypes)

# -----------------------------
# Memory Usage
# -----------------------------
print("\n5. Memory Usage")
print(df.memory_usage(deep=True))

print("\nInspection Completed Successfully.")