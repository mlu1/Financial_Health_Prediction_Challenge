"""
Data loading module for SME Financial Health Index Challenge
Handles loading and basic exploration of training and test data
"""

import pandas as pd
import numpy as np
from typing import Tuple
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_train_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load training data from CSV file
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    df = pd.read_csv(config.TRAIN_FILE)
    
    # Separate features and target
    X = df.drop(columns=[config.TARGET, config.ID_COLUMN])
    y = df[config.TARGET]
    
    return X, y


def load_test_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load test data from CSV file
    
    Returns:
        Tuple of (features DataFrame, ID Series)
    """
    df = pd.read_csv(config.TEST_FILE)
    
    X = df.drop(columns=[config.ID_COLUMN])
    ids = df[config.ID_COLUMN]
    
    return X, ids


def load_variable_definitions() -> pd.DataFrame:
    """
    Load variable definitions
    
    Returns:
        DataFrame with variable names and descriptions
    """
    return pd.read_csv(config.VARIABLE_DEFS_FILE)


def load_sample_submission() -> pd.DataFrame:
    """
    Load sample submission format
    
    Returns:
        DataFrame with expected submission format
    """
    return pd.read_csv(config.SAMPLE_SUBMISSION_FILE)


def get_data_info(X: pd.DataFrame, y: pd.Series = None) -> None:
    """
    Print summary statistics about the data
    
    Args:
        X: Features DataFrame
        y: Target Series (optional)
    """
    print("="*60)
    print("DATA INFORMATION")
    print("="*60)
    print(f"\nShape: {X.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    
    if y is not None:
        print(f"\nTarget Distribution:")
        print(y.value_counts(normalize=True))
    
    print(f"\nMissing Values:")
    missing = X.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values")
    
    print(f"\nData Types:")
    print(X.dtypes.value_counts())
    print("="*60)


if __name__ == "__main__":
    # Test data loading
    X_train, y_train = load_train_data()
    get_data_info(X_train, y_train)
    
    X_test, test_ids = load_test_data()
    print(f"\nTest data shape: {X_test.shape}")
    print(f"Test samples: {len(test_ids)}")

