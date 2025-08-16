# Bank Customer Churn Prediction (AIML)

## Overview
This project predicts whether a banking customer will exit (churn) using machine learning. The objective is to identify at-risk customers and surface the factors that drive churn so teams can take proactive retention actions. The analysis uses a structured customer dataset (`Churn.csv`) with demographics, account activity, and product usage features.

---

## Goals
- Predict if a customer will churn (Exited = 1) or remain (Exited = 0)
- Identify key drivers of churn to inform retention strategy
- Provide an interpretable, reproducible ML workflow

---

## Methodology

### 1) Data Preparation
- Loaded tabular data from `Churn.csv`
- Split into train/test with `train_test_split`
- Scaled numeric features (e.g., `CreditScore`, `Age`, `Balance`) using `StandardScaler`
- Encoded categorical features (e.g., `Geography`, `Gender`)
- Removed/leaked identifiers (e.g., `RowNumber`, `CustomerId`, `Surname`) from modeling

### 2) Modeling
- Trained supervised classifiers for binary churn prediction
- Performed hyperparameter search with `GridSearchCV`
- Selected final model based on validation performance

### 3) Evaluation and Insights
- Reported accuracy, precision, recall, F1-score, and confusion matrix on the test set
- Interpreted top features that separate churned vs. retained customers

---

## Repository Structure
```
AIML-Bank-Churn-Prediction/
├── bank_churn_prediction.ipynb # Main notebook (rename from Bank Churn Prediction Project.ipynb)
├── Churn.csv # Dataset (structured banking customer data)
└── README.md # Project documentation
```

---

## Dataset
The dataset includes common banking customer attributes such as:
- Demographics: Geography, Gender, Age
- Account and product usage: Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember
- Financials: CreditScore, EstimatedSalary
- Target: Exited (1 = churned, 0 = retained)

---

## Results
- Final model achieved strong performance on the hold-out test set  
- Include your final metrics here (e.g., Accuracy, Precision/Recall/F1, Confusion Matrix)  
- Most influential features typically include activity and product usage signals (e.g., `IsActiveMember`, `NumOfProducts`), along with `Age` and `CreditScore`
