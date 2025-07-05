import yaml
import joblib
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.features.preprocessing import get_preprocessor

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
    X_transformed = resolve_path(config["data"]["X"])
    y= resolve_path(config["data"]["y"])
    model_output_path = resolve_path(config["data"]["pipeline_path"])

    X_df = pd.read_csv(X_transformed)
    y_df = pd.read_csv(y)

    preprocessor = joblib.load(model_output_path)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        # Build the full pipeline, which includes preprocessing and the classifier
        pipeline = Pipeline([
            ("preprocessor", preprocessor),  # Preprocessing step (includes added features)
            ("classifier", LogisticRegression(max_iter=1000))  # Estimator (Logistic Regression)
        ])

        pipeline.fit(X_train, y_train)

        model_output_path = resolve_path(config["data"]["pipeline_path"])
        joblib.dump(pipeline, model_output_path)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_artifact(model_output_path)
        mlflow.sklearn.log_model(pipeline, artifact_path=model_output_path)

        print("✅ Model trained and logged to MLflow.")

if __name__ == "__main__":
    train()
