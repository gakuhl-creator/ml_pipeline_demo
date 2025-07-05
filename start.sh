#!/bin/bash
source venv/bin/activate
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow db init
airflow scheduler &
airflow webserver --port 8080 &
mlflow ui --port 5000 &
