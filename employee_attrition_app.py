import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# ===== Load Model & Data =====
df = pd.read_csv("employee_attrition.csv")
model = joblib.load("attrition_model.pkl")  # your trained RandomForest

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("📊 Employee Attrition Predictor")

# ===== Sidebar Input =====
st.sidebar.header("Enter Employee Details")
age = st.sidebar.number_input("Age", min_value=18, max_value=65, value=30)
salary = st.sidebar.number_input("Salary", min_value=20000, max_value=200000, value=50000)
years = st.sidebar.number_input("Years at Company", min_value=0, max_value=40, value=2)
jobrole = st.sidebar.selectbox("Job Role", df['JobRole'].unique())

# Encode categorical input same as training
jobrole_encoded = pd.factorize(df['JobRole'])[0][0]  # simple example

# Prepare input
input_df = pd.DataFrame({
    'Age':[age],
    'Salary':[salary],
    'YearsAtCompany':[years],
    'JobRole':[jobrole_encoded]
})

# ===== Predict =====
if st.button("Predict Attrition"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"This employee is likely to **Leave** (Probability: {prob:.2f})")
    else:
        st.success(f"This employee is likely to **Stay** (Probability: {1-prob:.2f})")

# ===== Show EDA Plots =====
st.subheader("📈 Attrition Distribution")
st.image("eda_dashboard.png", use_column_width=True)