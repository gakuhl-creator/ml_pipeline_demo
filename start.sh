#!/bin/bash

# Exit on error
set -e

# Activate virtual environment
source venv/bin/activate

# Set Airflow home
export AIRFLOW_HOME=$(pwd)/airflow_home

# Initialize Airflow DB only if not already initialized
if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
  echo "Initializing Airflow DB..."
  airflow db init
fi

# Start Airflow scheduler
echo "Starting Airflow scheduler..."
airflow scheduler &

# Start Airflow webserver
echo "Starting Airflow webserver on port 8080..."
airflow webserver --port 8080 &

# Start MLflow UI
echo "Starting MLflow UI on port 5000..."
mlflow ui --port 5000 &

# Start FastAPI server
echo "Starting FastAPI server on port 8000..."
uvicorn api.serve_model:app --reload --port 8000
