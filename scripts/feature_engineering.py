"""
NASA Turbofan Engine Dataset Feature Engineering Module

This script engineers features for the NASA Turbofan Engine Dataset to improve predictive maintenance models.
It creates moving averages and rate-of-change features for sensor readings.

Usage:
    python feature_engineering.py --processed_data_path data/processed --features_path data/features
"""
import os
import pandas as pd
import numpy as np
import argparse
import logging

def engineer_features(processed_data_path, features_path):
    """
    Create engineered features for the model
    
    Args:
        processed_data_path: Path to processed data
        features_path: Path to save engineered features
    
    Returns:
        train_with_features: DataFrame with engineered features for training
        test_with_features: DataFrame with engineered features for testing
    """
    logging.info("Starting feature engineering...")
    
    # Load processed data
    train_df = pd.read_csv(os.path.join(processed_data_path, 'train_processed.csv'))
    test_df = pd.read_csv(os.path.join(processed_data_path, 'test_processed.csv'))
    
    logging.info(f"Loaded processed data: train shape {train_df.shape}, test shape {test_df.shape}")
    
    # Feature engineering
    # 1. Calculate moving averages for sensor readings
    sensor_cols = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
                  'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
                  'PCNfR_dmd', 'W31', 'W32']
    
    # Function to calculate moving averages for an engine
    def add_moving_average(df, engine_id, window_size=5):
        engine_data = df[df['engine_id'] == engine_id].copy()
        
        for col in sensor_cols:
            engine_data[f'{col}_ma{window_size}'] = engine_data[col].rolling(
                window=window_size, min_periods=1).mean()
        
        return engine_data
    
    # Apply moving averages to each engine
    logging.info("Creating moving average features...")
    train_with_features = pd.DataFrame()
    for engine_id in train_df['engine_id'].unique():
        engine_data = add_moving_average(train_df, engine_id)
        train_with_features = pd.concat([train_with_features, engine_data])
    
    test_with_features = pd.DataFrame()
    for engine_id in test_df['engine_id'].unique():
        engine_data = add_moving_average(test_df, engine_id)
        test_with_features = pd.concat([test_with_features, engine_data])
    
    # 2. Add features for rate of change (first derivative)
    logging.info("Creating rate-of-change features...")
    for col in sensor_cols:
        train_with_features[f'{col}_rate'] = train_with_features.groupby('engine_id')[col].diff()
        test_with_features[f'{col}_rate'] = test_with_features.groupby('engine_id')[col].diff()
    
    # Fill NaN values from diff operation - using the updated pandas method
    train_with_features = train_with_features.fillna(0)
    test_with_features = test_with_features.fillna(0)
    
    # Create directory if it doesn't exist
    os.makedirs(features_path, exist_ok=True)
    
    # Save engineered features
    train_with_features.to_csv(os.path.join(features_path, 'train_features.csv'), index=False)
    test_with_features.to_csv(os.path.join(features_path, 'test_features.csv'), index=False)
    
    logging.info(f"Engineered features saved to {features_path}")
    
    return train_with_features, test_with_features

def main():
    parser = argparse.ArgumentParser(description='Engineer features for NASA Turbofan Engine Dataset')
    parser.add_argument('--processed_data_path', type=str, default='data/processed', 
                        help='Path to processed data directory')
    parser.add_argument('--features_path', type=str, default='data/features', 
                        help='Path to save engineered features')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Starting feature engineering with processed data path: %s', args.processed_data_path)

    train_df, test_df = engineer_features(args.processed_data_path, args.features_path)

    # Log feature statistics and validation information
    logging.info('Feature engineering completed.')
    logging.info('Train data with features shape: %s', train_df.shape)
    logging.info('Test data with features shape: %s', test_df.shape)
    
    # Log number of features created
    num_features = train_df.shape[1] - 2  # Subtract engine_id and RUL columns
    logging.info('Number of features created: %d', num_features)
    
    # Log memory usage
    train_memory = train_df.memory_usage(deep=True).sum() / (1024 * 1024)
    test_memory = test_df.memory_usage(deep=True).sum() / (1024 * 1024)
    logging.info('Memory usage - Train: %.2f MB, Test: %.2f MB', train_memory, test_memory)

if __name__ == '__main__':
    main()
