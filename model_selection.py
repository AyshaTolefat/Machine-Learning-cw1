import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("output/train_data.csv")
X = df.drop(columns=["outcome"])
y = df["outcome"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": (LinearRegression(), {}),
    "Ridge": (Ridge(), {'alpha': [0.01, 0.1, 1, 10, 100]}),
    "Lasso": (Lasso(), {'alpha': [0.0001, 0.001, 0.01, 0.1]}),
    "Decision Tree": (DecisionTreeRegressor(random_state=42), {'max_depth': [5, 10, 15, 20], 'min_samples_leaf': [2, 5, 10], 'min_samples_split': [1, 3, 5]}),
    "Random Forest": (RandomForestRegressor(random_state=42), {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 25], 'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10]}),
    "XGBoost": (XGBRegressor(objective="reg:squarederror", random_state=42), {'n_estimators': [50, 100, 200, 250], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1], 'subsample': [0.8, 1.0], 'colsample_bytree':[0.8, 1.0]}),
    "Gradient Boosting": (GradientBoostingRegressor(random_state=42), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1]}),
    "AdaBoost": (AdaBoostRegressor(random_state=42), {'n_estimators': [50, 100, 200], 'learning_rate': [0.05, 0.1]})
}

best_model, best_score = None, -np.inf
results = []
for name, (model, params) in models.items():
    print(f"\nTraining {name}...")
    if params:
        search = RandomizedSearchCV(model, params, cv=5, scoring='r2', n_iter=10, n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print(f"Best params for {name}: {search.best_params_}")
    else:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        print(f"{name} Avg CV R²: {scores.mean():.4f}")
        model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    r2, mae, rmse = r2_score(y_val, y_pred), mean_absolute_error(y_val, y_pred), np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"{name} -> R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    results.append((name, r2, mae, rmse))
    if r2 > best_score:
        best_score, best_model = r2, model

joblib.dump(best_model, "output/best_model.pkl")
print(f"Best model saved with R² = {best_score:.4f}")

# Plot Performance
df_results = pd.DataFrame(results, columns=["Model", "R²", "MAE", "RMSE"]).set_index("Model")
df_results.plot(kind="bar", figsize=(14, 7))
plt.title("Model Performance Comparison")
plt.tight_layout()
plt.savefig("output/model_performance.png")
plt.show()

if hasattr(best_model, "feature_importances_"):
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(best_model.feature_importances_)), best_model.feature_importances_)
    plt.xticks(range(len(best_model.feature_importances_)), X.columns, rotation=90)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("output/feature_importance.png")
    plt.show()

