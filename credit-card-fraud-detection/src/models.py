"""
Models Module

Anomaly detection models for fraud detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def train_isolation_forest(X: pd.DataFrame, contamination: float, 
                           n_estimators: int = 100, random_state: int = 42) -> dict:
    """
    Train an Isolation Forest model for anomaly detection.
    
    Isolation Forest isolates anomalies by randomly selecting features
    and splitting values. Anomalies are easier to isolate, requiring
    fewer splits on average.
    
    Parameters
    ----------
    X : pd.DataFrame
        Scaled features
    contamination : float
        Expected proportion of anomalies (0 to 0.5)
    n_estimators : int
        Number of trees in the forest
    random_state : int
        Random state for reproducibility
    
    Returns
    -------
    dict
        Dictionary containing:
        - model: fitted IsolationForest
        - predictions: binary predictions (1 = anomaly, 0 = normal)
        - scores: anomaly scores (more negative = more anomalous)
    """
    print(f"\n[MODEL] Training Isolation Forest...")
    print(f"        - n_estimators: {n_estimators}")
    print(f"        - contamination: {contamination:.5f}")
    
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples='auto',
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X)
    
    # Get predictions (-1 = anomaly in sklearn, convert to 1 = anomaly)
    predictions = model.predict(X)
    predictions = np.where(predictions == -1, 1, 0)
    
    # Get anomaly scores (more negative = more anomalous)
    scores = model.decision_function(X)
    
    n_anomalies = predictions.sum()
    print(f"[MODEL] Isolation Forest detected {n_anomalies:,} anomalies "
          f"({n_anomalies / len(predictions) * 100:.3f}%)")
    
    return {
        'model': model,
        'predictions': predictions,
        'scores': scores
    }


def train_lof(X: pd.DataFrame, contamination: float, 
              n_neighbors: int = 20) -> dict:
    """
    Train a Local Outlier Factor model for anomaly detection.
    
    LOF compares the local density of each point to its neighbors.
    Points in less dense regions than their neighbors are considered
    outliers.
    
    Parameters
    ----------
    X : pd.DataFrame
        Scaled features
    contamination : float
        Expected proportion of anomalies (0 to 0.5)
    n_neighbors : int
        Number of neighbors for density estimation
    
    Returns
    -------
    dict
        Dictionary containing:
        - model: fitted LocalOutlierFactor
        - predictions: binary predictions (1 = anomaly, 0 = normal)
        - scores: anomaly scores (more negative = more anomalous)
    """
    print(f"\n[MODEL] Training Local Outlier Factor...")
    print(f"        - n_neighbors: {n_neighbors}")
    print(f"        - contamination: {contamination:.5f}")
    
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        metric='euclidean',
        n_jobs=-1
    )
    
    # LOF uses fit_predict (novelty=False by default)
    predictions = model.fit_predict(X)
    predictions = np.where(predictions == -1, 1, 0)
    
    # Get anomaly scores
    scores = model.negative_outlier_factor_
    
    n_anomalies = predictions.sum()
    print(f"[MODEL] LOF detected {n_anomalies:,} anomalies "
          f"({n_anomalies / len(predictions) * 100:.3f}%)")
    
    return {
        'model': model,
        'predictions': predictions,
        'scores': scores
    }


def train_all_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train all anomaly detection models.
    
    Uses the actual fraud rate as the contamination parameter.
    
    Parameters
    ----------
    X : pd.DataFrame
        Scaled features
    y : pd.Series
        Target variable (for calculating contamination rate)
    
    Returns
    -------
    dict
        Dictionary with results for each model
    """
    print("\n" + "=" * 60)
    print("TRAINING ANOMALY DETECTION MODELS")
    print("=" * 60)
    
    # Use actual fraud rate as contamination
    contamination = y.sum() / len(y)
    print(f"\nContamination rate (fraud ratio): {contamination:.5f} ({contamination * 100:.3f}%)")
    
    # Train models
    iso_results = train_isolation_forest(X, contamination)
    lof_results = train_lof(X, contamination)
    
    # Summary
    print("\n" + "-" * 60)
    print("TRAINING SUMMARY")
    print("-" * 60)
    print(f"{'Model':<25} {'Anomalies Detected':>20}")
    print("-" * 45)
    print(f"{'Isolation Forest':<25} {iso_results['predictions'].sum():>20,}")
    print(f"{'LOF':<25} {lof_results['predictions'].sum():>20,}")
    print(f"{'Actual Frauds':<25} {y.sum():>20,}")
    print("=" * 60)
    
    return {
        'isolation_forest': iso_results,
        'lof': lof_results,
        'contamination': contamination
    }
