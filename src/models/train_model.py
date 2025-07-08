import joblib
import yaml
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.features.preprocess import get_preprocessor  # Import the function that returns the preprocessor

def resolve_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", relative_path))

def load_config():
    config_path = resolve_path("src/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def train():
    # Load the configuration
    config = load_config()
    clean_data_path = resolve_path(config["data"]["clean_path"])
    model_output_path = resolve_path(config["data"]["pipeline_path"])
    X_test_path = resolve_path(config["data"]["X_test_path"])
    y_test_path = resolve_path(config["data"]["y_test_path"])

    # Load clean data (this is the untransformed data)
    df = pd.read_csv(clean_data_path)

    # Load the preprocessor from preprocess.py
    preprocessor = get_preprocessor()  # This returns the preprocessor pipeline

    # Split the data into features (X) and target (y)
    target_column = config["target_column"]
    y = df[target_column].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=[target_column])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the pipeline with both preprocessing and model fitting steps
    pipeline = Pipeline([
        ("preprocessor", preprocessor),  # Preprocessing step (includes added features)
        ("classifier", LogisticRegression(max_iter=1000))  # Estimator (Logistic Regression)
    ])

    # Fit the pipeline with training data (includes preprocessing)
    pipeline.fit(X_train, y_train)

    # Extract the feature names after preprocessing
    # Get column names after the OneHotEncoder transformation
    # categorical_columns = preprocessor.transformers_[1][1].named_steps['encoder'].get_feature_names_out([
    #     "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    #     "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    #     "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    #     "Contract", "PaperlessBilling", "PaymentMethod"
    # ])

    # # Combine numeric and categorical features
    # all_features = list(["tenure", "MonthlyCharges", "TotalCharges"]) + list(categorical_columns)

    # all_features_dic = {li: type(li) for li in all_features}

    # print(all_features_dic)

    # # Print all features
    # print("Features required by the API:", all_features)

    # Save the trained pipeline (including the model and the preprocessor)
    joblib.dump(pipeline, model_output_path)

    # Save the train_test_splitstest files for evaluation
    X_test.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    print(f"Model trained and saved to {model_output_path}")

if __name__ == "__main__":
    train()
