"""
NASA Turbofan Engine Model Evaluation Script

This script evaluates trained models and compares their performance.
Results are logged using MLflow.

Usage:
    python evaluate.py --model_path models/saved --features_path data/features
"""
import os
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_test_data(features_path):
    """Load the test dataset"""
    test_path = os.path.join(features_path, 'test_features.csv')
    logging.info(f"Loading test data from {test_path}")
    
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(['engine_id', 'cycle', 'RUL'], axis=1)
    y_test = test_df['RUL']
    
    return X_test, y_test, test_df

def load_models(model_path):
    """Load all trained models from the model directory"""
    logging.info(f"Loading models from {model_path}")
    
    models = {}
    for filename in os.listdir(model_path):
        if filename.endswith('.joblib'):
            model_name = filename.replace('_model.joblib', '')
            model_file = os.path.join(model_path, filename)
            models[model_name] = joblib.load(model_file)
            logging.info(f"Loaded model: {model_name}")
    
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate all models on the test data and compare their performance"""
    # Check if experiment exists before creating it
    client = MlflowClient()
    experiment_name = "turbofan_model_comparison"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        # Create experiment if it doesn't exist
        experiment_id = client.create_experiment(experiment_name)
        logging.info(f"Created new experiment: {experiment_name} with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        logging.info(f"Using existing experiment: {experiment_name} with ID: {experiment_id}")
    
    # Set the experiment after checking/creating it
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="model_comparison"):
        results = {}
        
        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            results[model_name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                f"{model_name}_rmse": rmse,
                f"{model_name}_mae": mae,
                f"{model_name}_r2": r2
            })
            
            logging.info(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        # Create comparison plots
        plot_predictions(y_test, results)
        
        # Determine best model
        best_model, best_rmse = min(
            [(model_name, results[model_name]['rmse']) for model_name in results],
            key=lambda x: x[1]
        )
        
        logging.info(f"Best model: {best_model} with RMSE: {best_rmse:.4f}")
        mlflow.log_param("best_model", best_model)
        mlflow.log_metric("best_model_rmse", best_rmse)
        
        return results, best_model

def plot_predictions(y_true, results):
    """Create plots comparing model predictions"""
    plt.figure(figsize=(12, 8))
    
    # Plot actual RUL values
    plt.plot(y_true.values[:100], label='Actual RUL', linewidth=2)
    
    # Plot predictions from each model
    for model_name, metrics in results.items():
        plt.plot(metrics['predictions'][:100], label=f'{model_name} Predictions', linestyle='--')
    
    plt.title('RUL Predictions Comparison (First 100 samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Remaining Useful Life (cycles)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_path = 'model_comparison_plot.png'
    plt.savefig(plot_path)
    
    # Log the plot to MLflow
    mlflow.log_artifact(plot_path)
    
    logging.info(f"Comparison plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate RUL prediction models')
    parser.add_argument('--model_path', type=str, default='models/saved', 
                        help='Path to saved models directory')
    parser.add_argument('--features_path', type=str, default='data/features', 
                        help='Path to engineered features directory')
    args = parser.parse_args()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Load test data
    X_test, y_test, test_df = load_test_data(args.features_path)
    
    # Load models
    models = load_models(args.model_path)
    
    # Evaluate models
    results, best_model = evaluate_models(models, X_test, y_test)
    
    logging.info(f"Model evaluation completed successfully")
    logging.info(f"Best model: {best_model}")

if __name__ == "__main__":
    main()
