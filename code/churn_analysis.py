# Customer Churn Prediction in Telecommunications
# This script loads, cleans, explores and models customer churn data.

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. CREATE OUTPUT FOLDER
# This makes sure the figures folder exists before saving graphs.
# Create figures folder if it does not exist

os.makedirs("figures", exist_ok=True)

# 2. Load dataset
# The dataset is the Telco Customer Churn dataset from Kaggle.
# It contains customer service, contract and billing information.

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)
print(df.head())
print(df.info())

# 3. CHECK FOR MISSING VALUES
# This helps identify data quality problems before modelling.
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 4. DATA CLEANING
# TotalCharges is stored as text, so it must be converted to numeric.
# Invalid values are changed to missing values and then removed.

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

print("\nDataset shape after cleaning:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 5. EXPLORATORY DATA ANALYSIS GRAPH 1
# This graph shows how many customers churned and stayed.

plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.savefig("figures/churn_distribution.png", bbox_inches="tight")
plt.close()

# 6. EXPLORATORY DATA ANALYSIS GRAPH 2
# This graph compares contract type with churn.
# It helps show whether monthly contract customers churn more.

plt.figure(figsize=(7, 4))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Contract Type Compared with Churn")
plt.xticks(rotation=30)
plt.savefig("figures/contract_vs_churn.png", bbox_inches="tight")
plt.close()

# 7. EXPLORATORY DATA ANALYSIS GRAPH 3
# This graph compares monthly charges between churned and retained customers.

plt.figure(figsize=(6, 4))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges Compared with Churn")
plt.savefig("figures/monthly_charges_vs_churn.png", bbox_inches="tight")
plt.close()

print("\nGraphs saved in figures folder")


# 8. PREPARE DATA FOR MACHINE LEARNING
# Machine learning models need numbers, not text.
# One-hot encoding converts text categories into numeric columns.

data = df.copy()
data = data.drop("customerID", axis=1)

# Convert all text columns into numeric columns using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

X = data.drop("Churn_Yes", axis=1)
y = data["Churn_Yes"]

# 9. SPLIT DATA INTO TRAINING AND TESTING SETS
# 80% of the data is used for training.
# 20% is used for testing model performance.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 10. CREATE THREE MACHINE LEARNING MODELS
# Logistic Regression: simple and interpretable.
# Decision Tree: rule-based and easy to understand.
# Random Forest: stronger ensemble model using many trees.

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# 11. TRAIN AND EVALUATE MODELS
# Each model is trained on the training data and tested on unseen data.
# Accuracy and classification report are used for comparison.

print("\nMODEL RESULTS")
print("--------------------------------")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nModel:", name)
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(classification_report(y_test, predictions))

# 12. END OF SCRIPT
# The outputs are printed in the terminal and graphs are saved.
print("\nAnalysis complete.")
