"""
Main FastAPI application for RUL prediction serving.

This module creates a FastAPI application to serve the turbofan engine RUL model.
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import mlflow
import logging
import os
from api.endpoints import health, prediction
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from api.metrics import set_model_version

# Configure logging for better debugging and tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI from environment variable or default if not set
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

# Create FastAPI app with meta information
app = FastAPI(
    title="Turbofan Engine RUL Prediction API",
    description="API for predicting Remaining Useful Life of turbofan engines",
    version="1.0.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include different API routes/routers
app.include_router(health.router, tags=["Health"])
app.include_router(prediction.router, tags=["Prediction"])

@app.on_event("startup")
async def startup_event():
    """Runs when the API starts up."""
    logger.info("Starting up RUL Prediction API")
    try:
        # Create MLflow client to interact with the MLflow server
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        experiments = client.list_experiments()
        logger.info(f"Connected to MLflow, found {len(experiments)} experiments")
        
        # Retrieve the latest production version of the model
        model_name = "turbofan_rul_model"
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            version = versions[0].version
            # Update Prometheus metrics with the model version
            set_model_version(model_name, version)
            logger.info(f"Model {model_name} version {version} is deployed")
    except Exception as e:
        # Log any connection issues with MLflow
        logger.error(f"Failed to connect to MLflow: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint returning API welcome message."""
    return {
        "message": "Welcome to the Turbofan Engine RUL Prediction API",
        "docs": "/docs",
    }

@app.get("/metrics")
async def metrics():
    """Endpoint to expose Prometheus metrics."""
    # Generate and return the latest metrics data from Prometheus
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Run the application using uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
