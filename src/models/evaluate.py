import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)
import joblib
from sklearn.model_selection import train_test_split

from src.utils import load_config, resolve_path

def evaluate():
    """
    Evaluate model on test set and log metrics/artifacts to MLflow.
    """
    config = load_config()
    paths = config["paths"]

    X = pd.read_csv(resolve_path(paths["X"]))
    y = pd.read_csv(resolve_path(paths["y"])).squeeze()
    model = joblib.load(resolve_path(paths["model"]))

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="Yes")
    recall = recall_score(y_test, y_pred, pos_label="Yes")
    f1 = f1_score(y_test, y_pred, pos_label="Yes")

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(nested=True):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Plot and log confusion matrix
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.tight_layout()
        cm_path = resolve_path(paths["confusion_matrix"])
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)

        print("✅ Evaluation complete. Metrics and confusion matrix logged.")
