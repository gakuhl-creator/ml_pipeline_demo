import os
import pandas as pd
import joblib
import yaml
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

def resolve_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", relative_path))

def load_config(path="src/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def evaluate():
    config = load_config()
    experiment_name = config["mlflow"]["experiment_name"]
    mlflow_tracking_uri = config["mlflow"]["tracking_uri"]


    # Load test data and model
    X_test = pd.read_csv(resolve_path(config["data"]["X_test_path"]))
    y_test = pd.read_csv(resolve_path(config["data"]["y_test_path"])).squeeze()
    model = joblib.load(resolve_path(config["data"]["pipeline_path"]))

    # Predict
    y_pred = model.predict(X_test)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Calculate error metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Assemble the metrics we're going to write into a collection
    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    params = {
        "max_iter": 1000
    }

    experiment = mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Initiate the MLflow run context
    with mlflow.start_run() as run:
        # Log the parameters used for the model fit
        mlflow.log_params(params)

        # Log the error metrics that were calculated during validation
        mlflow.log_metrics(metrics)

        # Log an instance of the trained model for later use
        mlflow.sklearn.log_model(sk_model=model, input_example=X_test, name=experiment_name)

    output_path = resolve_path(config["evaluation"]["output_path"])
    plt.savefig(output_path)
    plt.close()
