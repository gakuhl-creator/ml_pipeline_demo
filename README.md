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
git clone https://github.com/YOUR_USERNAME/gusto-ml-churn-pipeline.git
cd gusto-ml-churn-pipeline
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Airflow and MLflow

```bash
chmod +x start.sh
./start.sh
```

> This starts:
> - Airflow scheduler at `http://localhost:8080`
> - MLflow UI at `http://localhost:5000`

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
mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: churn-prediction

paths:
  X: data/X.csv
  y: data/y.csv
  model: data/model.pkl
  confusion_matrix: data/conf_matrix.png
```

Update paths and experiment names here for easy portability.

---

## 🛠️ Airflow Usage

From the UI at `http://localhost:8080`:

1. Turn on `churn_pipeline`
2. Trigger DAG manually
3. Monitor task logs and results

Or use CLI:

```bash
airflow dags trigger churn_pipeline
```

---

## ✅ Next Steps

- Track data versions and schema drift
- Deploy Model as API using Flask or FastAPI
- Containerize with Docker

---

## 🧾 License

MIT License
