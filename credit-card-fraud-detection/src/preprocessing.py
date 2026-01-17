"""
Preprocessing Module

Functions for data preprocessing and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def separate_features_target(df: pd.DataFrame) -> tuple:
    """
    Separate features (X) and target (y) from the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset with 'Class' column as target
    
    Returns
    -------
    tuple
        (X, y) where X is features DataFrame and y is target Series
    """
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print(f"[PREPROCESSING] Features: {X.shape[1]} columns")
    print(f"[PREPROCESSING] Target: {y.shape[0]} values ({y.sum()} frauds)")
    
    return X, y


def scale_features(X: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Standardize specified columns using StandardScaler.
    
    By default, scales 'Time' and 'Amount' columns since V1-V28
    are already scaled from PCA transformation.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame
    columns : list, optional
        Columns to scale. Defaults to ['Time', 'Amount']
    
    Returns
    -------
    pd.DataFrame
        DataFrame with scaled columns
    """
    if columns is None:
        columns = ['Time', 'Amount']
    
    X_scaled = X.copy()
    scaler = StandardScaler()
    
    for col in columns:
        if col in X_scaled.columns:
            X_scaled[col] = scaler.fit_transform(X[[col]])
            print(f"[PREPROCESSING] Scaled '{col}': "
                  f"mean {X[col].mean():.2f} → {X_scaled[col].mean():.2f}, "
                  f"std {X[col].std():.2f} → {X_scaled[col].std():.2f}")
    
    return X_scaled


def create_2d_projection(X: pd.DataFrame, random_state: int = 42) -> tuple:
    """
    Create a 2D PCA projection of the features for visualization.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame (should be scaled)
    random_state : int
        Random state for reproducibility
    
    Returns
    -------
    tuple
        (X_2d, pca) where X_2d is the 2D projection and pca is the fitted PCA object
    """
    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X)
    
    variance_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"[PREPROCESSING] 2D PCA projection created")
    print(f"[PREPROCESSING] Variance explained: {variance_explained:.1f}%")
    
    return X_2d, pca


def preprocess_pipeline(df: pd.DataFrame) -> dict:
    """
    Run the complete preprocessing pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
    
    Returns
    -------
    dict
        Dictionary containing:
        - X: original features
        - X_scaled: scaled features
        - y: target variable
        - X_2d: 2D projection
        - pca: fitted PCA object
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Separate features and target
    X, y = separate_features_target(df)
    
    # Scale Time and Amount
    print("\nScaling features...")
    X_scaled = scale_features(X)
    
    # Create 2D projection for visualization
    print("\nCreating 2D projection...")
    X_2d, pca = create_2d_projection(X_scaled)
    
    print("=" * 60)
    
    return {
        'X': X,
        'X_scaled': X_scaled,
        'y': y,
        'X_2d': X_2d,
        'pca': pca
    }
