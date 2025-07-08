import os
import pandas as pd
import joblib
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def resolve_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", relative_path))

def load_config(path="src/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def evaluate():
    config = load_config()

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

    output_path = resolve_path(config["evaluation"]["output_path"])
    plt.savefig(output_path)
    plt.close()
