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

# Create figures folder if it does not exist
os.makedirs("figures", exist_ok=True)

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)
print(df.head())
print(df.info())
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Clean TotalCharges column
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

print("\nDataset shape after cleaning:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Graph 1: Churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.savefig("figures/churn_distribution.png", bbox_inches="tight")
plt.close()

# Graph 2: Contract type vs churn
plt.figure(figsize=(7, 4))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Contract Type Compared with Churn")
plt.xticks(rotation=30)
plt.savefig("figures/contract_vs_churn.png", bbox_inches="tight")
plt.close()

# Graph 3: Monthly charges vs churn
plt.figure(figsize=(6, 4))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges Compared with Churn")
plt.savefig("figures/monthly_charges_vs_churn.png", bbox_inches="tight")
plt.close()

print("\nGraphs saved in figures folder")

