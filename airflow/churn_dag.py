import sys, os

# Add the project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
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
    t2 = PythonOperator(task_id='preprocess', python_callable=preprocess)
    t3 = PythonOperator(ask_id="train",
    python_callable=train,
    op_kwargs={
        "n_estimators": 200,
        "max_depth": 8
    },
    dag=dag,
)
    t4 = PythonOperator(task_id='evaluate', python_callable=evaluate)

    t1 >> t2 >> t3 >> t4
