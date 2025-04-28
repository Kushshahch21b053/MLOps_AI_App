"""
Prediction endpoint to serve the RUL prediction model.
"""
from fastapi import APIRouter, HTTPException, Depends, Request
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from api.models import PredictionInput, PredictionResponse
from api.metrics import PREDICTIONS, PREDICTION_VALUE, PREDICTION_ERROR, track_requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

def load_model():
    """Load the model from MLflow model registry."""
    try:
        model_name = "turbofan_rul_model"
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
        logger.info(f"Successfully loaded model {model_name}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")

@router.post("/predict", response_model=PredictionResponse)
@track_requests()
async def predict(input_data: PredictionInput, request: Request):
    """Endpoint to predict Remaining Useful Life (RUL) for a turbofan engine."""
    try:
        # Track prediction attempt
        PREDICTIONS.inc()
        
        # Load the model
        model = load_model()
        
        # Create the initial DataFrame with ALL columns from the original dataset
        data = {
            # These aren't used for prediction but might be needed for column ordering
            'engine_id': [input_data.engine_id],
            'cycle': [0],  # Placeholder
            
            # Settings (the first 3 values from settings array)
            'setting1': [input_data.settings[0] if len(input_data.settings) > 0 else 0],
            'setting2': [input_data.settings[1] if len(input_data.settings) > 1 else 0],
            'setting3': [input_data.settings[2] if len(input_data.settings) > 2 else 0]
        }
        
        # Sensor values - map directly to the correct column names
        sensor_cols = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
                       'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 
                       'PCNfR_dmd', 'W31', 'W32']
        
        for i, col in enumerate(sensor_cols):
            if i < len(input_data.sensors):
                data[col] = [input_data.sensors[i]]
            else:
                data[col] = [0]  # Default if not enough inputs
        
        # Create base dataframe with all original columns
        df = pd.DataFrame(data)
        
        # Add moving average columns with same values
        for col in sensor_cols:
            df[f'{col}_ma5'] = df[col]
        
        # Add rate columns
        for col in sensor_cols:
            df[f'{col}_rate'] = [0]
            
        # Drop engine_id and cycle - the model doesn't need these
        prediction_df = df.drop(['engine_id', 'cycle'], axis=1)
        
        # Make prediction
        rul_prediction = model.predict(prediction_df)[0]
        
        # Track prediction value in histogram
        PREDICTION_VALUE.observe(rul_prediction)
        
        # Return result
        return {
            "engine_id": input_data.engine_id,
            "remaining_useful_life": float(rul_prediction),
            "confidence_interval": {
                "lower": float(max(0, rul_prediction - 10)),
                "upper": float(rul_prediction + 10)
            },
            "prediction_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        # Track prediction error
        PREDICTION_ERROR.inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
