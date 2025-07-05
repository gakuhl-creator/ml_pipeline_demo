import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.utils import load_config, resolve_path

def train(n_estimators=100, max_depth=None):
    """
    Train a Random Forest model on churn dataset,
    save model locally, and log to MLflow.
    """
    config = load_config()
    paths = config["paths"]

    X = pd.read_csv(resolve_path(paths["X"]))
    y = pd.read_csv(resolve_path(paths["y"])).squeeze()

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42
        )
        model.fit(X_train, y_train)

        model_path = resolve_path(paths["model"])
        joblib.dump(model, model_path)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("✅ Model trained and logged to MLflow.")
