"""
Pydantic models for API request and response validation.
This module defines the expected structures for API inputs and outputs.
"""

from pydantic import BaseModel, Field  # Base class and field customization for Pydantic models
from typing import List, Dict, Optional  # Type hints for list, dictionary, and optional fields

class PredictionInput(BaseModel):
    """Model for prediction input data received by the API."""
    # Unique engine identifier
    engine_id: int = Field(..., description="Engine ID")
    # List of operational settings parameters
    settings: List[float] = Field(..., description="Operational settings")
    # List of sensor measurements from the engine
    sensors: List[float] = Field(..., description="Sensor measurements")
    
    class Config:
        # Example provided to assist with client-side validations and documentation
        schema_extra = {
            "example": {
                "engine_id": 1,
                "settings": [0.0, -0.0002, 100.0],
                "sensors": [518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61, 554.36, 2388.06, 
                            9046.19, 1.3, 47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388]
            }
        }

class HealthResponse(BaseModel):
    """Model for API health check response."""
    # API operational status
    status: str
    # Application version details
    version: str
    # Optional model version if applicable
    model_version: Optional[str] = None

class PredictionResponse(BaseModel):
    """Model for sending prediction responses back to the client."""
    # Engine ID corresponding to the request
    engine_id: int
    # Predicted remaining useful life of the engine
    remaining_useful_life: float
    # Optional detailed confidence interval for the prediction
    confidence_interval: Optional[Dict[str, float]] = None
    # Timestamp when the prediction was made
    prediction_time: str
