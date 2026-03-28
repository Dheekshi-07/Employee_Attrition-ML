Employee Attrition Prediction

Project Overview

This project is an **industry-oriented machine learning system** designed to predict employee attrition. Employee attrition is a critical problem for organizations as it leads to high costs and workforce instability. Using HR and performance data, this system identifies employees likely to leave and provides insights for retention strategies.



Dataset

- File: `employee_attrition.csv`
- Contains HR-related features such as:
  - Age, Salary, Job Role, Years at Company, Department, etc.
- Includes the target variable: `Attrition` (Yes/No)

> Note: A clean and preprocessed dataset is used to improve model performance.

---


Project Features

1. **Exploratory Data Analysis (EDA)**
   - Attrition distribution
   - Attrition by Job Role
   - Correlation heatmap
   - Box plots (Years at Company vs Attrition, Salary vs Attrition)
2. **Feature Engineering**
   - Experience Level (`Low` / `High`)
   - Salary Level (`Low` / `High`)
3. **Machine Learning Models**
   - Random Forest Classifier (main model)
   - Logistic Regression / Decision Tree (optional)
4. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrix
   - Top contributing features for attrition
5. **Streamlit Web App**
   - Interactive dashboard for EDA
   - User input for predictions (Age, Salary, Years at Company, Job Role)
   - Displays prediction result and probability




Tools & Libraries

- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit



How to Run the Project



1. Run the ML & EDA script

```bash
python3 employee_attrition.py

This script performs data cleaning, EDA, feature engineering, model training, and saves the trained model (attrition_model.pkl) and plots.

2. Run the Streamlit App

streamlit run employee_attrition_app.py

streamlit run employee_attrition_app.py

Open the link in your browser (usually http://localhost:8501)
Explore EDA graphs and make predictions using the sidebar inputs

Insights & Key Findings

Employees with low experience tend to leave more frequently
Employees with low salary have higher attrition
Certain job roles are at higher risk of attrition
Feature importance highlights Salary, Years at Company, and Job Role as key drivers

Folder Structure

Project-4-Employee_Attrition/
│
├── employee_attrition.csv
├── employee_attrition.py
├── employee_attrition_app.py
├── attrition_model.pkl
├── attrition_distribution.png
├── attrition_by_jobrole.png
├── correlation_heatmap.png
├── years_attrition_boxplot.png
├── salary_vs_attrition.png
├── top_features.png
└── screenshots/   (terminal + output screenshots)

Author
Name: Dheekshika
GitHub: https://github.com/Dheekshi-07

License
This project is for educational purposes.

