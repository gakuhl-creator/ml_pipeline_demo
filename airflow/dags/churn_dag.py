import sys
import os

# This file lives in: airflow_home/
# Your project root is: ../ (ml_pipeline_demo)
DAGS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DAGS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.data_ingest.load_data import load_data
from src.features.preprocess import preprocess
from src.models.train_model import train
from src.models.evaluate import evaluate

default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

with DAG(dag_id='churn_pipeline',
         description='Train and evaluate churn prediction model',
         tags=['ml', 'churn', 'mlflow'],
         default_args=default_args,
         schedule=None,
         catchup=False) as dag:

    t1 = PythonOperator(task_id='load_data', python_callable=load_data)
    t2 = PythonOperator(task_id='preprocess', python_callable=preprocess, do_xcom_push=False)
    t3 = PythonOperator(task_id="train", python_callable=train)
    t4 = PythonOperator(task_id='evaluate', python_callable=evaluate)

    t1 >> t2 >> t3 >> t4
