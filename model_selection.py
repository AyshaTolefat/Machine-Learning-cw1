import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load processed training data
df = pd.read_csv("output/train_data.csv")

# Split features & target
X = df.drop(columns=["outcome"])
y = df["outcome"]

# Train/Validation Split (80/20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models with hyperparameter tuning
models = {
    "Linear Regression": (LinearRegression(), {}),
    "Decision Tree": (DecisionTreeRegressor(random_state=42), {
        'max_depth': [5, 10, 20, 30, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5]
    }),
    "Random Forest": (RandomForestRegressor(random_state=42), {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }),
    "XGBoost": (XGBRegressor(objective="reg:squarederror", random_state=42), {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9, 12],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    })
}

# Evaluate models with RandomizedSearchCV and cross-validation
best_model = None
best_score = -np.inf
results = []

for name, (model, params) in models.items():
    print(f"\nTraining {name}...")
    if params:
        random_search = RandomizedSearchCV(model, params, cv=5, scoring='r2', n_iter=20, n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_
        print(f"Best hyperparameters for {name}: {random_search.best_params_}")
    else:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        print(f"{name} Cross-Validation R² Scores: {scores}")
        print(f"{name} Average CV R²: {scores.mean():.4f}")
        model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    print(f"{name} Evaluation:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}\n")

    results.append((name, r2, mae, rmse))

    if r2 > best_score:
        best_score = r2
        best_model = model

# Save best model
BEST_MODEL_PATH = "output/best_model.pkl"
joblib.dump(best_model, BEST_MODEL_PATH)
print(f"Best model saved: {BEST_MODEL_PATH} (R² Score = {best_score:.4f})")

# Plot model performance
results_df = pd.DataFrame(results, columns=["Model", "R² Score", "MAE", "RMSE"])
results_df.set_index("Model", inplace=True)

plt.figure(figsize=(12, 6))
results_df[["R² Score", "MAE", "RMSE"]].plot(kind="bar", figsize=(12, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("output/model_performance.png")
plt.show()
print("Model performance plot saved.")

# Feature Importance for tree-based models
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feature_names = X.columns
    sorted_indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.bar(range(len(importances)), importances[sorted_indices])
    plt.xticks(range(len(importances)), feature_names[sorted_indices], rotation=90)
    plt.title("Feature Importance of Best Model")
    plt.tight_layout()
    plt.savefig("output/feature_importance.png")
    plt.show()
    print("Feature importance plot saved.")
else:
    print("Best model does not support feature importance.")

