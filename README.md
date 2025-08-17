# Bank Churn Classification

## Overview
This project builds a machine learning pipeline to predict customer churn in a retail bank. The dataset contains demographic, account, and transactional information, which is used to classify whether a customer will leave the bank.  

The goal is to provide insights into churn drivers and evaluate different ML algorithms for predictive performance.

---

## Dataset
The dataset includes:
- Customer demographics (age, gender, geography)
- Account information (balance, number of products, credit score, tenure)
- Activity-related variables (estimated salary, activity status)
- Target: **Exited** (1 = churned, 0 = retained)

---

## Workflow
1. **Data Preprocessing**
   - Handled missing values and outliers  
   - Encoded categorical variables (OneHotEncoding for Geography, Gender)  
   - Scaled numerical features with StandardScaler  

2. **Exploratory Data Analysis (EDA)**
   - Distribution plots of churned vs retained customers  
   - Correlation analysis of numerical features  
   - Feature importance visualizations  

3. **Modeling**
   - Logistic Regression  
   - Random Forest  
   - XGBoost  
   - Neural Network (Keras Sequential model)

4. **Evaluation Metrics**
   - Accuracy  
   - Precision, Recall, F1-Score  
   - ROC-AUC  

---

## Results
- **Logistic Regression**: Baseline model, accuracy ~79%  
- **Random Forest**: Accuracy ~86%, ROC-AUC ~0.90  
- **XGBoost**: Accuracy ~87%, ROC-AUC ~0.91  
- **Neural Network**: Accuracy ~85%, ROC-AUC ~0.89  

**Best Model:** XGBoost achieved the highest overall performance with a balance between precision and recall, making it the recommended model for deployment.  

---

## Key Insights
- Customers with lower credit scores and shorter tenure are more likely to churn.  
- Geography and gender showed moderate influence on churn likelihood.  
- High account balance alone does not guarantee retention — tenure and product count play stronger roles.  

---

## Repository Structure

Bank-Churn-Prediction/
├── Bank Churn Prediction Project.ipynb # Main notebook
├── Churn.csv # Dataset (Twitter US Airline Sentiment)
└── README.md # Project documentation

---

## Technologies Used
- Python (Pandas, NumPy, Scikit-learn)  
- XGBoost  
- Keras / TensorFlow  
- Matplotlib & Seaborn (visualization)  
