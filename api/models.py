"""
Pydantic models for API request and response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PredictionInput(BaseModel):
    """Model for prediction input data."""
    engine_id: int = Field(..., description="Engine ID")
    settings: List[float] = Field(..., description="Operational settings")
    sensors: List[float] = Field(..., description="Sensor measurements")
    
    class Config:
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
    status: str
    version: str
    model_version: Optional[str] = None

class PredictionResponse(BaseModel):
    """Model for prediction response."""
    engine_id: int
    remaining_useful_life: float
    confidence_interval: Optional[Dict[str, float]] = None
    prediction_time: str
