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
    """Get the version of the deployed model."""
    try:
        client = MlflowClient()
        model_name = "turbofan_rul_model"
        latest_version = client.get_latest_versions(model_name, stages=["Production"])
        if latest_version:
            return latest_version[0].version
        return "Not found"
    except Exception as e:
        return f"Error: {str(e)}"

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model_version": get_model_version()
    }

@router.get("/readiness", response_model=HealthResponse)
async def readiness_check():
    """Readiness check to verify if the model is loaded and ready to serve predictions."""
    model_version = get_model_version()
    model_status = "ready" if model_version not in ["Not found", None] else "not ready"
    
    return {
        "status": model_status,
        "version": "1.0.0",
        "model_version": model_version
    }
