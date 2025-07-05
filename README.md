# Telco Customer Churn Prediction

This project predicts customer churn using machine learning techniques based on historical telecom usage data. It was built as a practice exercise in classification modeling, with an emphasis on maximizing recall for identifying at-risk customers.

---

## 🔍 Project Overview

Customer churn (when users stop using a service) is a major challenge for telecom companies. Identifying potential churners early allows for proactive retention efforts.

This project:
- Cleans and prepares raw customer data
- Builds and trains a Random Forest model
- Tunes the threshold to prioritize **recall** (catching churners)
- Evaluates performance with classification metrics and visualizations

---

## 📁 Data

The dataset comes from [Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/) and includes customer demographics, account information, and service usage.

Key features include:
- `tenure`, `MonthlyCharges`, `TotalCharges`
- Service types (`PhoneService`, `InternetService`, etc.)
- Contract type, payment method, and more

The target variable is `Churn` (Yes/No).

---

## 🧪 Modeling Pipeline

1. **Load & Inspect**: Understand raw structure
2. **Clean & Prepare**: Handle nulls, encode categoricals, scale numerical features
3. **Feature Engineer**: Normalize numeric features
4. **Train-test Split**: random_state=42 (the ultimate answer, right?)
5. **Train a Random Forest model**: 100 trees
6. **Evaluate**: Precision, recall, F1-score, confusion matrix.
7. **Save the Trained Model**



---

## ✅ Results

- **Recall (churn class)**: **75%**  
- **Precision (churn class)**: 51%  
- **Accuracy**: 74%

🎯 The model effectively catches 3 out of 4 churners with a reasonable precision tradeoff — ideal for customer retention campaigns.

---

## 📂 Project Structure

telco-churn-predictor/
├── data/
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│ └── X_test.csv
│ └── X_train.csv
│ └── Y_test.csv
│ └── Y_train.csv
├── models/
│ └── random_forest_churn.pkl
├── notebooks/
│ ├── telco_churn_predictor.ipynb
├── README.md
└── requirements.txt


---

## 🛠 How to Run

1. Clone the repo
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Run notebooks in order (Jupyter or VSCode)

---

📄 View notebook outputs:
- [Model Training & Tuning (HTML)](notebooks/html_exports/telco_churn_predictor.html)

---

## 🚀 Future Work

- Hyperparameter tuning
- SMOTE / resampling to boost minority class further
- Model explainability via SHAP or feature permutation
- Deployment via Flask or Streamlit
- Host on a Cloud

---

## 📌 Author

Guillermo Kuhl

All Rights Reserved.

This project was completed as part of a self-directed ML portfolio. Built using Python, scikit-learn, pandas, matplotlib, seaborn, and Jupyter.

