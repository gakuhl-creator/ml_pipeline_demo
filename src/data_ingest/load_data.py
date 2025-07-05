import pandas as pd
import os
import yaml

def resolve_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", relative_path))

def load_data():
    # Load config
    config_path = resolve_path("src/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_path = resolve_path(config["data"]["raw_path"])
    clean_path = resolve_path(config["data"]["clean_path"])

    # Load raw data
    df = pd.read_csv(raw_path)

    # Clean up any non-numeric values that may exist in numeric columns (convert blank spaces or text to NaN)
    numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce') 

    # Drop rows with missing values in key columns
    df = df.dropna(subset=["tenure", "MonthlyCharges", "TotalCharges", config["target_column"]])

    df = df.drop(columns=['customerID'])

    # Save cleaned data
    df.to_csv(clean_path, index=False)

    print(f"Cleaned data saved to {clean_path}")
