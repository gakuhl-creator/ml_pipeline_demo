import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def resolve_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", relative_path))

def load_config():
    config_path = resolve_path("src/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def predict():
    # Load the saved model pipeline (preprocessor + classifier)
    config = load_config()
    model_path = resolve_path(config["data"]["pipeline_path"])
    pipeline = joblib.load(model_path)

    # Load new data (for prediction)
    X_new = pd.read_csv("path/to/your/new_data.csv")  # Replace with your actual data path

    # Make predictions using the loaded pipeline
    predictions = pipeline.predict(X_new)

    print(f"Predictions: {predictions}")

if __name__ == "__main__":
    predict()
