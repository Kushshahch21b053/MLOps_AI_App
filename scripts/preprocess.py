"""
NASA Turbofan Engine Dataset Preprocessing Module

This script preprocesses the NASA Turbofan Engine Dataset for predictive maintenance.
It calculates Remaining Useful Life (RUL) values for both training and test datasets.

Usage:
    python preprocess.py --raw_data_path data/raw --processed_data_path data/processed
"""
import os
import pandas as pd
import numpy as np
import argparse
import logging

def load_dataset(data_path):
    """
    Load the NASA Turbofan Engine Dataset
    """
    # Column names for the dataset
    columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 'T2', 
               'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
               'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
               'PCNfR_dmd', 'W31', 'W32']
    
    # Load training data
    train_df = pd.read_csv(
        os.path.join(data_path, 'train_FD001.txt'), 
        sep=r'\s+', 
        header=None, 
        names=columns
    )
    
    # Load test data
    test_df = pd.read_csv(
        os.path.join(data_path, 'test_FD001.txt'), 
        sep=r'\s+', 
        header=None, 
        names=columns
    )
    
    # Load RUL data
    rul_df = pd.read_csv(
        os.path.join(data_path, 'RUL_FD001.txt'),
        sep=r'\s+',
        header=None,
        names=['RUL']
    )
    
    return train_df, test_df, rul_df

def preprocess_data(raw_data_path, processed_data_path):
    """
    Load and preprocess data
    
    Args:
        raw_data_path: Path to raw data
        processed_data_path: Path to save processed data
    """
    logging.info("Starting data preprocessing...")
    
    # Load data
    train_df, test_df, rul_df = load_dataset(raw_data_path)
    
    # Calculate RUL for training data
    # Group by engine_id and calculate max cycle for each engine
    max_cycles = train_df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    
    # Merge with the original dataframe
    train_df = train_df.merge(max_cycles, on=['engine_id'], how='left')
    
    # Calculate RUL (max_cycle - current_cycle)
    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
    train_df.drop('max_cycle', axis=1, inplace=True)
    
    # Add RUL to test data
    # First, get the max cycle for each engine
    test_max_cycles = test_df.groupby('engine_id')['cycle'].max().reset_index()
    test_max_cycles.columns = ['engine_id', 'max_cycle']
    
    # Merge max cycles with test data
    test_df = test_df.merge(test_max_cycles, on=['engine_id'], how='left')
    
    # Add RUL column from the RUL dataframe
    test_with_rul = pd.DataFrame()
    for engine_id in test_df['engine_id'].unique():
        # Get max cycle for this engine
        max_cycle = test_df.loc[test_df['engine_id'] == engine_id, 'max_cycle'].iloc[0]
        
        # Get engine data
        engine_data = test_df[test_df['engine_id'] == engine_id].copy()
        
        # Get RUL for this engine
        rul_value = rul_df.loc[engine_id-1, 'RUL']
        
        # Calculate RUL for each cycle
        engine_data['RUL'] = rul_value + (max_cycle - engine_data['cycle'])
        
        # Append to the new dataframe
        test_with_rul = pd.concat([test_with_rul, engine_data])
    
    test_with_rul.drop('max_cycle', axis=1, inplace=True)
    
    # Handle any missing values
    train_df = train_df.ffill()
    test_with_rul = test_with_rul.ffill()
    
    # Create directory if it doesn't exist
    os.makedirs(processed_data_path, exist_ok=True)
    
    # Save processed data
    train_df.to_csv(os.path.join(processed_data_path, 'train_processed.csv'), index=False)
    test_with_rul.to_csv(os.path.join(processed_data_path, 'test_processed.csv'), index=False)
    
    logging.info(f"Processed data saved to {processed_data_path}")
    
    return train_df, test_with_rul

def main():
    parser = argparse.ArgumentParser(description='Preprocess NASA Turbofan Engine Dataset')
    parser.add_argument('--raw_data_path', type=str, default='data/raw', help='Path to raw data directory')
    parser.add_argument('--processed_data_path', type=str, default='data/processed', help='Path to save processed data')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Starting preprocessing with raw data path: %s', args.raw_data_path)

    train_df, test_df = preprocess_data(args.raw_data_path, args.processed_data_path)

    logging.info('Preprocessing completed. Processed train data shape: %s', train_df.shape)
    logging.info('Processed test data shape: %s', test_df.shape)
    logging.info('Train data statistics: \n%s', train_df['RUL'].describe())
    logging.info('Test data statistics: \n%s', test_df['RUL'].describe())

if __name__ == '__main__':
    main()
