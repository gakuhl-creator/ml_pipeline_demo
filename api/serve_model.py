import joblib
import pandas as pd
import yaml
import os
from fastapi import FastAPI
from pydantic import BaseModel

def resolve_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", relative_path))

def load_config():
    config_path = resolve_path("src/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# Load model pipeline (preprocessing + model)
config = load_config()
model_path = config["data"]["pipeline_path"]
model = joblib.load(model_path)

# Define request schema for raw input features (before preprocessing)
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
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

class Item(BaseModel):
    prediction: int
    prediction_english: str
    churn_probability: float

@app.post("/predict")
def predict(data: CustomerFeatures) -> Item:
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Ensure numeric fields are properly converted
    input_df["tenure"] = pd.to_numeric(input_df["tenure"], errors="coerce")
    input_df["MonthlyCharges"] = pd.to_numeric(input_df["MonthlyCharges"], errors="coerce")
    input_df["TotalCharges"] = pd.to_numeric(input_df["TotalCharges"], errors="coerce")

    # Predict using full pipeline
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]  # probability of class 1

    return Item(prediction=prediction, churn_probability=proba, prediction_english= "Churn" if prediction == 1 else "No Churn")
