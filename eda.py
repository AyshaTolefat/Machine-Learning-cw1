# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# File path to your dataset
DATA_PATH = "CW1_train.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Set a consistent style for visualizations
sns.set(style="whitegrid")

# Output folder for saving EDA visuals
OUTPUT_FOLDER = "output"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

### Basic Overview of the Dataset ###
print("\n--- Basic Information ---")
print(df.info())

print("\n--- First Five Rows ---")
print(df.head())

print("\n--- Summary Statistics ---")
print(df.describe())

# Save Summary Statistics to a file
df.describe().to_csv(os.path.join(OUTPUT_FOLDER, "summary_statistics.csv"))

# Check for missing values
print("\n--- Missing Values ---")
missing_values = df.isnull().sum()
print(missing_values)

# Save missing values summary
missing_values.to_csv(os.path.join(OUTPUT_FOLDER, "missing_values.csv"))

### Correlation Heatmap ###
numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(16, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap of Numerical Features")
plt.savefig(os.path.join(OUTPUT_FOLDER, "correlation_heatmap.png"))
plt.close()
print("Correlation heatmap saved.")

### Identifying Highly Correlated Features ###
correlation_threshold = 0.9
high_correlation_pairs = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname1 = correlation_matrix.columns[i]
            colname2 = correlation_matrix.columns[j]
            high_correlation_pairs.add((colname1, colname2))

print("\nHighly Correlated Features (above 0.9):")
print(high_correlation_pairs)

# Suggesting to drop one of the highly correlated pairs
features_to_drop = set(col2 for _, col2 in high_correlation_pairs)
print(f"Suggested Features to Drop: {features_to_drop}")

df = df.drop(columns=features_to_drop, errors="ignore")  # Ignore errors if columns are missing

### Target Variable Analysis ###
plt.figure(figsize=(8, 6))
sns.histplot(df['outcome'], kde=True, bins=30)
plt.title("Distribution of Target Variable ('outcome')")
plt.xlabel("Outcome")
plt.ylabel("Frequency")
plt.savefig(os.path.join(OUTPUT_FOLDER, "target_variable_distribution.png"))
plt.close()
print("Target variable distribution plot saved.")

### Categorical Variables Encoding (One-Hot Encoding) ###
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("\nOne-Hot Encoding categorical variables...")
one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_cats = one_hot_encoder.fit_transform(df[categorical_cols])

# Convert encoded categorical variables to DataFrame
encoded_df = pd.DataFrame(encoded_cats, columns=one_hot_encoder.get_feature_names_out(categorical_cols))

# Drop original categorical columns and merge encoded ones
df = df.drop(columns=categorical_cols)
df = pd.concat([df, encoded_df], axis=1)

print("Categorical variables encoded successfully.")

### Handling Missing Values ###
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

### Outlier Detection & Removal (Fixed!) ###
# Dynamically filter numerical columns that still exist
selected_numerical_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
selected_numerical_cols = [col for col in selected_numerical_cols if col in df.columns]  # Only keep existing columns

if selected_numerical_cols:
    print(f"\nApplying outlier detection on: {selected_numerical_cols}")
    Q1 = df[selected_numerical_cols].quantile(0.10)  # Relaxed from 25%
    Q3 = df[selected_numerical_cols].quantile(0.90)  # Relaxed from 75%
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Removing only the most extreme outliers
    df_cleaned = df[
        ~((df[selected_numerical_cols] < lower_bound) | (df[selected_numerical_cols] > upper_bound)).any(axis=1)
    ]
else:
    print("No numerical columns selected for outlier removal. Skipping...")
    df_cleaned = df

# Ensure data is not empty after outlier removal
if df_cleaned.shape[0] == 0:
    print("WARNING: All rows were removed after outlier removal. Adjusting threshold...")
    df_cleaned = df  # Revert to original dataset if too many points are lost

print(f"Data shape after outlier removal: {df_cleaned.shape}")

### Feature Scaling (Standardization) ###
scaler = StandardScaler()
df_cleaned.loc[:, 'price'] = df_cleaned['price'].astype(float)
df_cleaned.loc[:, numeric_columns] = scaler.fit_transform(df_cleaned[numeric_columns])


### Splitting Data for Model Training ###
X = df_cleaned.drop(columns=['outcome'])  # Features
y = df_cleaned['outcome']  # Target Variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed datasets
TRAIN_DATA_PATH = os.path.join(OUTPUT_FOLDER, "train_data.csv")
TEST_DATA_PATH = os.path.join(OUTPUT_FOLDER, "test_data.csv")

pd.concat([X_train, y_train], axis=1).to_csv(TRAIN_DATA_PATH, index=False)
pd.concat([X_test, y_test], axis=1).to_csv(TEST_DATA_PATH, index=False)

print(f"\nTrain data saved to {TRAIN_DATA_PATH}")
print(f"Test data saved to {TEST_DATA_PATH}")

