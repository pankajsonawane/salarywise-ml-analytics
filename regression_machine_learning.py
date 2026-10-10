
# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 2. DEFINE PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Salary_Data.csv"
MODEL_PATH = BASE_DIR / "linear_regression_model.pkl"

print("=" * 60)
print("SALARYWISE ML ANALYTICS - MODEL TRAINING")
print("=" * 60)


# ============================================================
# 3. LOAD DATASET
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Please keep Salary_Data.csv in the project folder."
    )

dataset = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print(f"Dataset shape: {dataset.shape}")
print("\nFirst 5 records:")
print(dataset.head())


# ============================================================
# 4. IDENTIFY INPUT (X) AND OUTPUT (y)
# ============================================================

if dataset.shape[1] < 2:
    raise ValueError(
        "Dataset must contain at least one input column "
        "and one target column."
    )

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
    raise ValueError("All input feature columns must be numeric.")

if not pd.api.types.is_numeric_dtype(y):
    raise ValueError("The target column must be numeric.")

if X.isnull().any().any() or y.isnull().any():
    raise ValueError(
        "Dataset contains missing values. Clean them before training."
    )

print("\nInput features:")
print(list(X.columns))

print("\nTarget variable:")
print(y.name)


# ============================================================
# 5. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 6. TRAIN LINEAR REGRESSION MODEL
# ============================================================

regressor = LinearRegression()

regressor.fit(X_train, y_train)

print("\nModel training completed successfully.")


# ============================================================
# 7. PREDICT TEST DATA
# ============================================================

y_pred = regressor.predict(X_test)

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Residual": y_test - y_pred
})

print("\nActual vs Predicted:")
print(comparison.round(2))


# ============================================================
# 8. VISUALIZE ACTUAL DATA AND REGRESSION LINE
# ============================================================

if X.shape[1] == 1:
    feature_name = X.columns[0]
    target_name = y.name

    plt.figure(figsize=(10, 6))

    plt.scatter(
        X_test[feature_name],
        y_test,
        color="red",
        label="Actual Test Data"
    )

    # Sort input values so the regression line displays correctly
    X_line = X.sort_values(by=feature_name)
    y_line = regressor.predict(X_line)

    plt.plot(
        X_line[feature_name],
        y_line,
        color="blue",
        label="Regression Line"
    )

    plt.title(f"{target_name} vs {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel(target_name)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print(
        "\nSkipping 2D regression plot because the dataset "
        "contains multiple input features."
    )


# ============================================================
# 9. REGRESSION COEFFICIENTS AND INTERCEPT
# ============================================================

print("\nModel Coefficients (Slopes):")
print(regressor.coef_)

print("\nModel Intercept:")
print(regressor.intercept_)

if X.shape[1] == 1:
    slope = regressor.coef_[0]
    intercept = regressor.intercept_

    print(
        f"\nRegression Equation: "
        f"Salary = {slope:.2f} * {X.columns[0]} "
        f"+ {intercept:.2f}"
    )

    for experience in [12, 20]:
        input_data = pd.DataFrame(
            {X.columns[0]: [experience]}
        )
        predicted_salary = regressor.predict(input_data)[0]

        print(
            f"Predicted salary for {experience} years "
            f"of experience: {predicted_salary:.2f}"
        )


# ============================================================
# 10. MODEL PERFORMANCE METRICS
# ============================================================

train_r2 = regressor.score(X_train, y_train)
test_r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Training R² Score: {train_r2:.4f}")
print(f"Testing R² Score : {test_r2:.4f}")
print(f"MAE              : {mae:.4f}")
print(f"MSE              : {mse:.4f}")
print(f"RMSE             : {rmse:.4f}")


# ============================================================
# 11. SSR, SSE, SST AND R-SQUARED
# ============================================================

# All calculations below use the TEST dataset only.

test_mean = np.mean(y_test)

# Regression Sum of Squares
SSR = np.sum((y_pred - test_mean) ** 2)

# Error Sum of Squares
SSE = np.sum((y_test.to_numpy() - y_pred) ** 2)

# Total Sum of Squares
SST = np.sum((y_test.to_numpy() - test_mean) ** 2)

# R² from sums of squares
if SST != 0:
    r_square = 1 - (SSE / SST)
else:
    r_square = np.nan

print("\n" + "=" * 60)
print("SUM OF SQUARES")
print("=" * 60)

print(f"SSR: {SSR:.4f}")
print(f"SSE: {SSE:.4f}")
print(f"SST: {SST:.4f}")
print(f"R² from SSE/SST: {r_square:.4f}")
print(f"R² from sklearn : {test_r2:.4f}")


# ============================================================
# 12. STATISTICAL ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL ANALYSIS")
print("=" * 60)

print("\nMean:")
print(dataset.mean(numeric_only=True))

print("\nMedian:")
print(dataset.median(numeric_only=True))

print("\nMode:")
print(dataset.mode().iloc[0])

print("\nVariance:")
print(dataset.var(numeric_only=True))

print("\nStandard Deviation:")
print(dataset.std(numeric_only=True))

print("\nSkewness:")
print(dataset.skew(numeric_only=True))


# ============================================================
# 13. SAVE TRAINED MODEL
# ============================================================

with MODEL_PATH.open("wb") as file:
    pickle.dump(regressor, file)

print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print(f"Model file: {MODEL_PATH}")
print(f"File exists: {MODEL_PATH.exists()}")
print(f"File size: {MODEL_PATH.stat().st_size:,} bytes")

print("\nTraining script completed successfully.")
