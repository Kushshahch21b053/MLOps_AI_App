"""
NASA Turbofan Engine Model Training Script

This script trains ML models to predict Remaining Useful Life (RUL) based on engineered features.
Models are tracked and logged using MLflow.

Usage:
    python train.py --features_path data/features --model_path models/saved
"""
import os
import argparse
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(features_path):
    """Load the engineered features dataset"""
    train_features_path = os.path.join(features_path, 'train_features.csv')
    test_features_path = os.path.join(features_path, 'test_features.csv')
    
    logging.info(f"Loading training features from {train_features_path}")
    train_df = pd.read_csv(train_features_path)
    
    logging.info(f"Loading test features from {test_features_path}")
    test_df = pd.read_csv(test_features_path)
    
    return train_df, test_df

def prepare_data(train_df, test_df, target_col='RUL'):
    """Prepare data for training by separating features and target"""
    # Drop columns that shouldn't be used as features
    drop_cols = ['engine_id', 'cycle', target_col]
    
    # Training data
    X_train = train_df.drop(drop_cols, axis=1)
    y_train = train_df[target_col]
    
    # Test data
    X_test = test_df.drop(drop_cols, axis=1)
    y_test = test_df[target_col]
    
    # Create validation split from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    logging.info(f"Prepared data - X_train: {X_train.shape}, y_train: {y_train.shape}")
    logging.info(f"Validation data - X_val: {X_val.shape}, y_val: {y_val.shape}")
    logging.info(f"Test data - X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_train, y_train, X_val, y_val, model_type="random_forest"):
    """Train a model with the given training data and validate it"""
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
    
    with mlflow.start_run(run_name=f"{model_type}_training"):
        logging.info(f"Training {model_type} model...")
        
        # Log dataset characteristics
        mlflow.log_param("training_samples", X_train.shape[0])
        mlflow.log_param("validation_samples", X_val.shape[0])
        mlflow.log_param("features", X_train.shape[1])
        
        if model_type == "random_forest":
            # Create pipeline with preprocessing and model
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(random_state=42))
            ])
            
            # Define hyperparameter grid
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__max_depth': [10, 20],
                'model__min_samples_split': [2, 5]
            }
            
            # Log hyperparameters to MLflow
            for param, values in param_grid.items():
                mlflow.log_param(f"grid_{param}", values)
            
        elif model_type == "gradient_boosting":
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingRegressor(random_state=42))
            ])
            
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__learning_rate': [0.01, 0.1],
                'model__max_depth': [3, 5]
            }
            
            for param, values in param_grid.items():
                mlflow.log_param(f"grid_{param}", values)
                
        elif model_type == "linear_regression":
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ])
            
            param_grid = {}  # No hyperparameters for linear regression
        
        # Perform grid search with cross-validation
        logging.info("Performing grid search with cross-validation...")
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, scoring='neg_mean_squared_error',
            verbose=1, n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Log best hyperparameters
        logging.info(f"Best hyperparameters: {grid_search.best_params_}")
        for param, value in grid_search.best_params_.items():
            clean_param = param.replace('model__', '')
            mlflow.log_param(clean_param, value)
        
        # Evaluate on validation set
        y_val_pred = best_model.predict(X_val)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        mae = mean_absolute_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)
        
        # Log metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        logging.info(f"Validation metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(best_model, "model")
        
        return best_model, grid_search.best_params_

def evaluate_model(model, X_test, y_test, model_type, model_path):
    """Evaluate model on test data and save the model"""
    logging.info(f"Evaluating {model_type} model on test data...")
    
    # Predict on test data
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    
    # Log metrics
    with mlflow.start_run(run_name=f"{model_type}_evaluation"):
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)
    
    logging.info(f"Test metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    # Save model to disk
    os.makedirs(model_path, exist_ok=True)
    model_file = os.path.join(model_path, f"{model_type}_model.joblib")
    joblib.dump(model, model_file)
    logging.info(f"Model saved to {model_file}")
    
    return rmse, mae, r2

def main():
    parser = argparse.ArgumentParser(description='Train models for RUL prediction')
    parser.add_argument('--features_path', type=str, default='data/features', 
                        help='Path to engineered features directory')
    parser.add_argument('--model_path', type=str, default='models/saved', 
                        help='Path to save trained models')
    parser.add_argument('--model_type', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting', 'linear_regression'],
                        help='Type of model to train')
    args = parser.parse_args()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Load data
    train_df, test_df = load_data(args.features_path)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(train_df, test_df)
    
    # Train model
    model, best_params = train_model(X_train, y_train, X_val, y_val, args.model_type)
    
    # Evaluate model
    rmse, mae, r2 = evaluate_model(model, X_test, y_test, args.model_type, args.model_path)
    
    logging.info(f"Model training and evaluation completed successfully")
    logging.info(f"Best params: {best_params}")
    logging.info(f"Test RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()
