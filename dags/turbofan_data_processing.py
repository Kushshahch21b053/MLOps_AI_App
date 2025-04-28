# dags/turbofan_data_processing.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

# Add the project directory to the Python path
sys.path.append('/opt/airflow')

# Import functions from scripts
from scripts.preprocess import preprocess_data
from scripts.feature_engineering import engineer_features

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'turbofan_data_processing',
    default_args=default_args,
    description='A pipeline to process NASA Turbofan Engine Dataset',
    schedule_interval=None,
    start_date=datetime(2025, 4, 26),
    catchup=False,
    tags=['turbofan', 'predictive_maintenance'],
)

def preprocess_data_task():
    raw_data_path = "/opt/airflow/data/raw"
    processed_data_path = "/opt/airflow/data/processed"
    preprocess_data(raw_data_path, processed_data_path)

def engineer_features_task():
    processed_data_path = "/opt/airflow/data/processed"
    features_path = "/opt/airflow/data/features"
    engineer_features(processed_data_path, features_path)

# Task to preprocess data
preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data_task,
    dag=dag,
)

# Task to engineer features
feature_engineering_task = PythonOperator(
    task_id='engineer_features',
    python_callable=engineer_features_task,
    dag=dag,
)

# Define the task dependencies
preprocess_task >> feature_engineering_task
