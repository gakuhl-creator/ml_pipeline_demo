import pandas as pd

def preprocess():
    df = pd.read_csv("../data/clean_input.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df = df.drop(columns=['customerID'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    X = pd.get_dummies(df.drop(columns=['Churn']))
    X.to_csv("../data/X.csv", index=False)
    y.to_csv("../data/y.csv", index=False)
    print("Preprocessing complete.")
