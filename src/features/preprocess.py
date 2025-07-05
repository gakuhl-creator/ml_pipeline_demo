import yaml
import os
import pandas as pd
import joblib
import re
from src.features.preprocessing import get_preprocessor

def resolve_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", relative_path))

def load_config():
    config_path = resolve_path("src/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def preprocess():
    # Load the configuration
    config = load_config()
    cleaned_data = resolve_path(config["data"]["clean_path"])
    output_dir = os.path.dirname(cleaned_data)
    model_output_path = resolve_path(config["data"]["pipeline_path"])
    target_column = config["target_column"]

    # Load raw data
    df = pd.read_csv(cleaned_data)

    # Get the preprocessor (this just defines the steps, but doesn't apply them yet)
    preprocessor = get_preprocessor()

    # Apply the transformations (this is where categorical_transformer is actually executed)
    X_transformed = preprocessor.fit_transform(df)

    # # Split into X (features) and y (target)
    y = X_transformed[target_column].map({'Yes': 1, 'No': 0})
    X = X_transformed.drop(columns=[target_column])

    # print("Shape of X:", X.shape)
    # print("Shape of y:", y.shape)

    # Save transformed data and target variable
    X.to_csv(os.path.join(output_dir, "X_transformed.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "y.csv"), index=False)

    # Save the preprocessing pipeline
    joblib.dump(preprocessor, model_output_path)

    print(f"Preprocessing complete. Transformed data saved to {output_dir}")
    print(f"Preprocessing pipeline saved to {model_output_path}")

if __name__ == "__main__":
    preprocess()
