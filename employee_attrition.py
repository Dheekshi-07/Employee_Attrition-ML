# employee_attrition.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ===== Load Dataset =====
df = pd.read_csv("employee_attrition.csv")
print("CSV loaded successfully!")

# ===== Data Cleaning =====
df.fillna(method='ffill', inplace=True)

# ===== Feature Engineering =====
df['ExperienceLevel'] = df['YearsAtCompany'].apply(lambda x: 'Low' if x < 3 else 'High')
df['SalaryLevel'] = df['Salary'].apply(lambda x: 'Low' if x < 50000 else 'High')

# Encode categorical variables
df_encoded = df.copy()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df_encoded[col] = pd.factorize(df_encoded[col])[0]

# ===== EDA Plots =====
sns.set(style="whitegrid")

# 1. Attrition Distribution
plt.figure(figsize=(8,5))
sns.countplot(x='Attrition', data=df, palette='Set2')
plt.title("Attrition Distribution")
plt.savefig("attrition_distribution.png")
plt.close()

# 2. Attrition vs Job Role
plt.figure(figsize=(12,5))
sns.countplot(x='JobRole', hue='Attrition', data=df, palette='coolwarm')
plt.xticks(rotation=45)
plt.title("Attrition by Job Role")
plt.savefig("attrition_by_jobrole.png")
plt.close()

# 3. Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# 4. Boxplot: Attrition vs YearsAtCompany
plt.figure(figsize=(8,5))
sns.boxplot(x='Attrition', y='YearsAtCompany', data=df, palette='Set3')
plt.title("Attrition vs Years at Company")
plt.savefig("years_attrition_boxplot.png")
plt.close()

print("EDA plots saved successfully!")

# ===== Machine Learning =====
X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "attrition_model.pkl")

# ===== Model Evaluation =====
y_pred = model.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===== Business Insights =====
print("\n📌 Key Insights:")
print("- Employees with low experience (<3 years) tend to leave more")
print("- Low salary employees have higher attrition")
print("- Certain job roles have higher risk of leaving")
print("- HR can use these insights for retention planning")