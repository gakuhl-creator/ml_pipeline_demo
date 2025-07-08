from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import joblib

def resolve_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", relative_path))

def load_config():
    config_path = resolve_path("src/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# Define numeric and categorical features
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

# Main preprocessing function to define and return the pipeline
def get_preprocessor():
    # Numeric feature transformer (Impute and Scale)
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Categorical feature transformer (Impute and Encode)
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='if_binary'))
    ])

    # Combine transformations into a full preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder='passthrough', verbose_feature_names_out=False)

    # Ensuring the output is a pandas DataFrame
    preprocessor.set_output(transform="pandas")

    return preprocessor

# If you want to save the preprocessor (e.g., to disk), you can call this function
def save_preprocessor(preprocessor, model_output_path):
    # Save the preprocessor to the specified path using joblib
    joblib.dump(preprocessor, model_output_path)
    print(f"Preprocessing pipeline saved to {model_output_path}")

if __name__ == "__main__":
    # If you want to save the preprocessor when running this script (for development purposes)
    config = load_config()
    model_output_path = resolve_path(config["data"]["pipeline_path"])
    preprocessor = get_preprocessor()
    save_preprocessor(preprocessor, model_output_path)
