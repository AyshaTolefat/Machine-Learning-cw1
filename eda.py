# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# File path to your dataset
DATA_PATH = "CW1_train.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Set a consistent style for visualizations
sns.set(style="whitegrid")

# Output folder for saving EDA visuals
OUTPUT_FOLDER = "output"

# Create output directory if it doesn't exist
import os
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

### Visualizing Correlations ###
# Select only numerical columns for correlation calculation
numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
correlation_matrix = df[numerical_cols].corr()

print("Generating correlation heatmap...")
# Generate the correlation heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap of Numerical Features")
plt.savefig(os.path.join(OUTPUT_FOLDER, "correlation_heatmap.png"))  # Save the figure
plt.close()  # Close the plot to avoid hanging
print("Correlation heatmap saved.")

### Target Variable Analysis ###
print("Generating target variable distribution plot...")
plt.figure(figsize=(8, 6))
sns.histplot(df['outcome'], kde=True, bins=30)
plt.title("Distribution of Target Variable ('outcome')")
plt.xlabel("outcome")
plt.ylabel("Frequency")
plt.savefig(os.path.join(OUTPUT_FOLDER, "target_variable_distribution.png"))  # Save the figure
plt.close()  # Close the plot
print("Target variable distribution plot saved.")

### Categorical Variables Analysis ###
print("Generating plots for categorical variables...")
# Identify categorical columns
categorical_cols = ['cut', 'color', 'clarity']  # Replace with actual categorical column names
for col in categorical_cols:
    print(f"Processing {col}...")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{col}_distribution.png"))  # Save the figure
    plt.close()  # Close the plot
print("Categorical variable plots saved.")

### Continuous Variables Analysis ###
print("Generating boxplots for continuous variables...")
# Boxplots for continuous variables vs target
continuous_cols = [
    col for col in df.columns if col not in categorical_cols + ['outcome']
]

# Limit the number of continuous variables processed to avoid hanging
continuous_cols = continuous_cols[:10]  # Adjust this number if needed

for col in continuous_cols:
    print(f"Processing {col} (boxplot)...")
    plt.figure(figsize=(10, 6))
    
    # Sample 500 rows to avoid hanging
    sample_df = df.sample(n=500, random_state=42)
    
    sns.boxplot(data=sample_df, x='outcome', y=col)
    plt.title(f"Boxplot of {col} vs outcome")
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{col}_boxplot.png"))  # Save the figure
    plt.close()  # Close the plot
print("Continuous variable boxplots saved.")

### Save the Cleaned Data ###
# Handle missing values (if any) - replace with mean as an example
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df_cleaned = df.copy()
df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].mean())

CLEANED_DATA_PATH = os.path.join(OUTPUT_FOLDER, "cleaned_data.csv")
df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
print(f"\nCleaned data saved to {CLEANED_DATA_PATH}")
