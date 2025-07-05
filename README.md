# Churn Prediction ML Pipeline

A production-grade machine learning pipeline for customer churn prediction using:

- **Airflow** for orchestration
- **MLflow** for experiment tracking
- **Scikit-learn** for modeling
- **Jupyter Notebooks** for analysis
- **WSL + Python virtualenv** for environment isolation

---

## 📦 Features

- Cleanly structured pipeline: `load → preprocess → train → evaluate`
- Tracked and reproducible ML runs via MLflow
- Modular and maintainable Python package layout
- Configurable through a centralized `config.yaml`
- Ready for local or cloud deployment

---

## 🚀 Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/gakuhl-creator/gusto-ml-churn-pipeline.git
cd gusto-ml-churn-pipeline
```

### 2. Create Virtual Environments

Python 3.11 is required

Here are the commands to generate the virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


### 3. Launch all the things

```bash
chmod +x start.sh
./start.sh
```

> This starts:
> - Airflow scheduler at `http://localhost:8080`
> - MLflow UI at `http://localhost:5000`
> - FastAPI with Swagger UI viewable at `http://localhost:8000/docs`


You will need to execute the following (paste in correct values):
```bash
airflow users create \
  --username admin \
  --firstname First \
  --lastname Last \
  --role Admin \
  --email admin@example.com \
  --password admin
```

---

## 🧠 Pipeline Tasks

| Task          | Description                                   |
|---------------|-----------------------------------------------|
| `load_data.py`   | Load and persist raw input data as CSV       |
| `preprocess.py`  | Clean and encode features                    |
| `train_model.py` | Train RandomForest model + log to MLflow     |
| `evaluate.py`    | Evaluate test data + log metrics + plot      |

Each task is independently runnable and integrated into an Airflow DAG (`churn_pipeline`).

---

## 📁 Folder Structure

```
ml_pipeline_demo/
├── airflow/                  # DAGs and Airflow config
│   └── churn_dag.py
├── data/                     # Input/output data
│   ├── X.csv
│   ├── y.csv
│   ├── model.pkl
│   └── conf_matrix.png
├── src/
│   ├── config.yaml           # Central config for paths + MLflow
│   ├── utils.py              # Config + path helpers
│   ├── data_ingest/
│   │   └── load_data.py
│   ├── features/
│   │   └── preprocess.py
│   └── models/
│       ├── train_model.py
│       └── evaluate.py
├── requirements.txt
├── start.sh
└── README.md
```

---

## 📊 MLflow Tracking

MLflow logs:

- Parameters: `n_estimators`, `max_depth`, etc.
- Metrics: `accuracy`, `precision`, `recall`, `f1`
- Artifacts: model pickle, confusion matrix image

Track at: [http://localhost:5000](http://localhost:5000)

---

## 🧬 Configuration (`config.yaml`)

```yaml
data:
  raw_path: data/clean_input.csv
  model_output_path: data/model_pipeline.pkl

model:
  pipeline_path: data/model_pipeline.pkl

mlflow:
  tracking_uri: http://127.0.0.1:5000
  experiment_name: churn_prediction

evaluation:
  output_path: data/conf_matrix.png

target_column: Churn

```

Update paths and experiment names here for easy portability.

---

## 🛠️ Airflow Usage

From the UI at `http://localhost:8080`:

1. Turn on `churn_pipeline`
2. Trigger DAG manually
3. Monitor task logs and results

Or use CLI (using ```venv```):

```bash
airflow dags trigger churn_pipeline
```

---

## Train the model manually from CLI

Make sure you are using `venv`

```python3
python3 -c "from src.models.train_model import train; train()"
```

---

## Run the API Server

You can visit the Swagger UI at ```http://localhost:8000/docs```


Send a JSON payload such as the following toward  ```/predict```

```JSON
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.35,
  "TotalCharges": 850.5
}
```

... and you will receive a response such the following:

```JSON
{
  "churn_prediction": "Yes"
}
```


## ✅ Next Steps

- Track data versions and schema drift
- Deploy Model as API using Flask or FastAPI
- Containerize with Docker

---

## 🧾 License

MIT License
