from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import yaml
import os

# Get absolute path to config.yaml inside the src directory
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "config.yaml")
CONFIG_PATH = os.path.abspath(CONFIG_PATH)

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Could not find config.yaml at {CONFIG_PATH}")

# Load config
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Load model pipeline (preprocessing + model)
model_path = config["data"]["pipeline_path"]
model = joblib.load(model_path)

# Define request schema
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# FastAPI app instance
app = FastAPI()

@app.post("/predict")
def predict(data: CustomerFeatures):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Convert numeric fields explicitly
    input_df["SeniorCitizen"] = pd.to_numeric(input_df["SeniorCitizen"], errors="coerce")
    input_df["tenure"] = pd.to_numeric(input_df["tenure"], errors="coerce")
    input_df["MonthlyCharges"] = pd.to_numeric(input_df["MonthlyCharges"], errors="coerce")
    input_df["TotalCharges"] = pd.to_numeric(input_df["TotalCharges"], errors="coerce")

    # Ensure all object fields are strings
    for col in input_df.columns:
        if input_df[col].dtype == object:
            input_df[col] = input_df[col].fillna("Unknown").astype(str)

    missing = set(model.feature_names_in_) - set(input_df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")


    # Predict using full pipeline
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]  # probability of class 1

    return {
        "prediction": prediction,
        "churn_probability": round(proba, 3)
    }
