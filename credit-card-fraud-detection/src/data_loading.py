"""
Data Loading Module

Functions for loading and exploring the credit card fraud dataset.
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the credit card transactions dataset.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file (creditcard.csv)
    
    Returns
    -------
    pd.DataFrame
        Loaded dataset
    
    Raises
    ------
    FileNotFoundError
        If the dataset file is not found
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[DATA] Loaded {len(df):,} transactions from {filepath}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] Could not find '{filepath}'")
        print("        Please download the dataset from Kaggle:")
        print("        https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        raise


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    
    Returns
    -------
    dict
        Dictionary containing dataset statistics
    """
    n_frauds = df['Class'].sum()
    n_legit = len(df) - n_frauds
    fraud_ratio = n_frauds / len(df) * 100
    
    info = {
        'n_rows': len(df),
        'n_columns': df.shape[1],
        'n_frauds': n_frauds,
        'n_legit': n_legit,
        'fraud_ratio': fraud_ratio,
        'memory_mb': df.memory_usage().sum() / 1e6,
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum()
    }
    
    return info


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    """
    info = get_dataset_info(df)
    
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape: {info['n_rows']:,} rows × {info['n_columns']} columns")
    print(f"Memory: {info['memory_mb']:.1f} MB")
    print(f"\nClass Distribution:")
    print(f"  - Legitimate: {info['n_legit']:,} ({100 - info['fraud_ratio']:.2f}%)")
    print(f"  - Fraudulent: {info['n_frauds']:,} ({info['fraud_ratio']:.3f}%)")
    print(f"\nRatio: 1 fraud for every {int(info['n_legit'] / info['n_frauds'])} legitimate transactions")
    print(f"\nData Quality:")
    print(f"  - Missing values: {info['missing_values']}")
    print(f"  - Duplicate rows: {info['duplicates']}")
    print("=" * 60)


def get_amount_statistics(df: pd.DataFrame) -> dict:
    """
    Get statistics about transaction amounts.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    
    Returns
    -------
    dict
        Dictionary with amount statistics for each class
    """
    legit = df[df['Class'] == 0]['Amount']
    fraud = df[df['Class'] == 1]['Amount']
    
    stats = {
        'overall': {
            'mean': df['Amount'].mean(),
            'median': df['Amount'].median(),
            'max': df['Amount'].max(),
            'std': df['Amount'].std()
        },
        'legitimate': {
            'mean': legit.mean(),
            'median': legit.median(),
            'max': legit.max()
        },
        'fraudulent': {
            'mean': fraud.mean(),
            'median': fraud.median(),
            'max': fraud.max()
        }
    }
    
    return stats


def get_top_correlations(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """
    Get features most correlated with the target (Class).
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    n : int
        Number of top correlations to return (positive and negative)
    
    Returns
    -------
    pd.Series
        Correlations with Class column, sorted by absolute value
    """
    correlations = df.corr()['Class'].drop('Class')
    return correlations.reindex(correlations.abs().sort_values(ascending=False).index)
