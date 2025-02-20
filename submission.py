import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the test dataset
test_data_path = "CW1_test.csv"
df_test = pd.read_csv(test_data_path)

# Load the trained model
model_path = "output/best_model.pkl"
best_model = joblib.load(model_path)

# Load the train dataset to ensure consistent feature preprocessing
train_data_path = "output/train_data.csv"  # Using processed train data
df_train = pd.read_csv(train_data_path)

# Identify categorical and numerical columns (from train set)
categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df_train.select_dtypes(include=["float64", "int64"]).columns.tolist()

# Drop 'outcome' from numerical_cols (not in test set)
numerical_cols = [col for col in numerical_cols if col != 'outcome']

# Manually Apply One-Hot Encoding (Same as in Training)
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(df_train[categorical_cols])  # Fit on train set

encoded_test = pd.DataFrame(encoder.transform(df_test[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))

# Drop original categorical columns and merge encoded ones
df_test = df_test.drop(columns=categorical_cols)
df_test = pd.concat([df_test, encoded_test], axis=1)

# Ensure test set has the same feature columns as train set
train_features = df_train.drop(columns=['outcome']).columns  # Train features used
missing_cols = set(train_features) - set(df_test.columns)  # Features missing in test
extra_cols = set(df_test.columns) - set(train_features)  # Extra test features

# Add missing columns with default value 0
for col in missing_cols:
    df_test[col] = 0

# Drop extra columns that are not in train set
df_test = df_test[train_features]  # Keep only relevant columns

# Manually Apply Standard Scaling (Same as in Training)
scaler = StandardScaler()
scaler.fit(df_train[numerical_cols])  # Fit on train set
df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])  # Transform test set

# Make Predictions
predictions = best_model.predict(df_test)

# Save predictions in the required format
submission_filename = "CW1_submission_21182996.csv"
pd.DataFrame({'yhat': predictions}).to_csv(submission_filename, index=False)

print(f"Submission file '{submission_filename}' has been generated successfully!")

