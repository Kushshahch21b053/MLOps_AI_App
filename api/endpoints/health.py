"""
Health check endpoints to monitor API and model status.
"""

from fastapi import APIRouter, Depends
import mlflow
from mlflow.tracking import MlflowClient
import os
from datetime import datetime
from api.models import HealthResponse

router = APIRouter()

def get_model_version():
    """
    Retrieve the version of the deployed model from MLflow.
    
    Returns:
        The latest production version of the model if available,
        otherwise 'Not found' or an error message.
    """
    try:
        client = MlflowClient()  # Initialize MLflow client to interact with model registry
        model_name = "turbofan_rul_model"  # Name of the model to check
        latest_version = client.get_latest_versions(model_name, stages=["Production"])
        if latest_version:
            return latest_version[0].version  # Return the version of the first model found in production
        return "Not found"  # Return not found if no production version exists
    except Exception as e:
        # Return error message in case of exception during MLflow operation
        return f"Error: {str(e)}"

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Endpoint for basic health check of the API.
    
    Returns:
        A dictionary containing API status, version, and model version.
    """
    return {
        "status": "healthy",  # API is operating normally
        "version": "1.0.0",   # Static version number
        "model_version": get_model_version()  # Current model version from MLflow
    }

@router.get("/readiness", response_model=HealthResponse)
async def readiness_check():
    """
    Readiness check endpoint to verify if the model is loaded and ready.
    
    Determines readiness by checking if there is a valid model version.
    Returns a status of 'ready' if the model is found, otherwise 'not ready'.
    """
    model_version = get_model_version()  # Get current model version
    model_status = "ready" if model_version not in ["Not found", None] else "not ready"
    
    return {
        "status": model_status,  # Readiness status based on model availability
        "version": "1.0.0",      # API version remains the same
        "model_version": model_version  # Current model version from MLflow
    }
