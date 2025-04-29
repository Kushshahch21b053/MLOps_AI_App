"""
Model Training DAG

This DAG trains a machine learning model for RUL prediction using the engineered features
from the data processing pipeline. It integrates with MLflow for experiment tracking.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow
from mlflow.tracking import MlflowClient
import sys
import os

# Add the project directory to the Python path
sys.path.append('/opt/airflow')

# Import the model training function
from models.train import load_data, prepare_data, train_model, evaluate_model

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
    'turbofan_model_training',
    default_args=default_args,
    description='Train ML models for RUL prediction',
    schedule_interval=None,  # Manual trigger for now
    start_date=datetime(2025, 4, 26),
    catchup=False,
    tags=['turbofan', 'predictive_maintenance', 'model_training'],
)

def train_random_forest_task():
    """Task to train a Random Forest model"""
    features_path = "/opt/airflow/data/features"
    model_path = "/opt/airflow/models/saved"
    
    # Load data
    train_df, test_df = load_data(features_path)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(train_df, test_df)
    
    # Train model
    model, _ = train_model(X_train, y_train, X_val, y_val, "random_forest")
    
    # Evaluate model
    rmse, mae, r2 = evaluate_model(model, X_test, y_test, "random_forest", model_path)
    
    return f"Random Forest model trained successfully. Test RMSE: {rmse:.4f}"

def train_gradient_boosting_task():
    """Task to train a Gradient Boosting model"""
    features_path = "/opt/airflow/data/features"
    model_path = "/opt/airflow/models/saved"
    
    # Load data
    train_df, test_df = load_data(features_path)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(train_df, test_df)
    
    # Train model
    model, _ = train_model(X_train, y_train, X_val, y_val, "gradient_boosting")
    
    # Evaluate model
    rmse, mae, r2 = evaluate_model(model, X_test, y_test, "gradient_boosting", model_path)
    
    return f"Gradient Boosting model trained successfully. Test RMSE: {rmse:.4f}"

# Function to register the best model in MLflow Model Registry
def register_best_model_fn():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("turbofan_model_comparison")
    if experiment is None:
        return "No experiment found to register model."
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"]  # Order by RMSE ascending (lower is better)
    )
    if not runs:
        return "No runs found to register model."
    best_run = runs[0]  # Fix: use the first (best) run from the list
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_details = mlflow.register_model(model_uri, "turbofan_rul_model")
    return f"Registered model: {model_details.name} version {model_details.version}"


# Task to train Random Forest model
train_rf_task = PythonOperator(
    task_id='train_random_forest',
    python_callable=train_random_forest_task,
    dag=dag,
)

# Task to train Gradient Boosting model
train_gb_task = PythonOperator(
    task_id='train_gradient_boosting',
    python_callable=train_gradient_boosting_task,
    dag=dag,
)

# Task to register the best model
register_model_task = PythonOperator(
    task_id='register_best_model',
    python_callable=register_best_model_fn,
    dag=dag,
)

# Define the task dependencies
[train_rf_task, train_gb_task] >> register_model_task
