import pandas as pd

def load_data():
    df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df.to_csv("../data/clean_input.csv", index=False)
    print("Data loaded.")
