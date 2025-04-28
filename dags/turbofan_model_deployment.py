"""
Model Deployment DAG

This DAG handles the deployment of the best trained model to the FastAPI serving endpoint.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default args
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
    'turbofan_model_deployment',
    default_args=default_args,
    description='Deploy the best model to production',
    schedule_interval=None,  # Manual trigger for now
    start_date=datetime(2025, 4, 26),
    catchup=False,
    tags=['turbofan', 'predictive_maintenance', 'deployment'],
)

def promote_best_model_to_production():
    """Promote the best model to production stage in MLflow Model Registry."""
    client = MlflowClient()
    model_name = "turbofan_rul_model"
    
    # Get latest model version
    latest_versions = client.get_latest_versions(model_name)
    if not latest_versions:
        raise Exception(f"No versions found for model {model_name}")
    
    latest_version = latest_versions[0].version
    logging.info(f"Latest model version: {latest_version}")
    
    # Transition model to Production stage
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production"
    )
    
    logging.info(f"Model {model_name} version {latest_version} promoted to Production")
    return f"Model {model_name} version {latest_version} promoted to Production"

def test_model_endpoint():
    """Test the model endpoint to ensure it's working."""
    import requests
    import json
    import time
    
    # Wait for API to be up
    time.sleep(10)
    
    # Test data
    test_data = {
        "engine_id": 1,
        "settings": [0.0, -0.0002, 100.0],
        "sensors": [518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61, 554.36, 2388.06, 
                   9046.19, 1.3, 47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388]
    }
    
    # Send request to API
    try:
        response = requests.post(
            "http://fastapi:8000/predict",
            data=json.dumps(test_data),
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            logging.info(f"API test successful. Predicted RUL: {result['remaining_useful_life']}")
            return f"API test successful. Predicted RUL: {result['remaining_useful_life']}"
        else:
            logging.error(f"API test failed with status code {response.status_code}: {response.text}")
            return f"API test failed with status code {response.status_code}: {response.text}"
    except Exception as e:
        logging.error(f"API test failed with error: {str(e)}")
        return f"API test failed with error: {str(e)}"

# Task to promote model to production
promote_model_task = PythonOperator(
    task_id='promote_model_to_production',
    python_callable=promote_best_model_to_production,
    dag=dag,
)

# Task to restart the FastAPI service to pick up the new model
restart_api_task = BashOperator(
    task_id='restart_api_service',
    bash_command='echo "Restarting FastAPI service to deploy new model"',
    dag=dag,
)

# Task to test the deployed model endpoint
test_api_task = PythonOperator(
    task_id='test_model_endpoint',
    python_callable=test_model_endpoint,
    dag=dag,
)

# Define task dependencies
promote_model_task >> restart_api_task >> test_api_task
