# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

### Target Variable Analysis ###
plt.figure(figsize=(8, 6))
sns.histplot(df['outcome'], kde=True, bins=30)
plt.title("Distribution of Target Variable ('outcome')")
plt.xlabel("Outcome")
plt.ylabel("Frequency")
plt.savefig(os.path.join(OUTPUT_FOLDER, "target_variable_distribution.png"))
plt.close()
print("Target variable distribution plot saved.")

### Categorical Variables Encoding ###
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    print(f"Processing {col}...")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{col}_distribution.png"))
    plt.close()

# Encode categorical variables for model training
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Categorical variables encoded.")

### Handling Missing Values ###
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df_cleaned = df.copy()
df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].mean())

### Outlier Detection & Removal ###
Q1 = df_cleaned[numeric_columns].quantile(0.25)
Q3 = df_cleaned[numeric_columns].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Removing outliers
df_cleaned = df_cleaned[~((df_cleaned[numeric_columns] < lower_bound) | (df_cleaned[numeric_columns] > upper_bound)).any(axis=1)]
print(f"Data shape after outlier removal: {df_cleaned.shape}")

### Feature Scaling (Standardization) ###
scaler = StandardScaler()
df_cleaned[numeric_columns] = scaler.fit_transform(df_cleaned[numeric_columns])

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

