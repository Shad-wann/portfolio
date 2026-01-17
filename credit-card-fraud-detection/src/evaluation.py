"""
Evaluation Module

Functions for evaluating anomaly detection models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                   scores: np.ndarray, model_name: str) -> dict:
    """
    Evaluate an anomaly detection model.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels (1 = fraud, 0 = legitimate)
    y_pred : np.ndarray
        Predicted labels (1 = anomaly, 0 = normal)
    scores : np.ndarray
        Anomaly scores (more negative = more anomalous)
    model_name : str
        Name of the model for display
    
    Returns
    -------
    dict
        Dictionary containing all evaluation metrics
    """
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Average Precision (area under precision-recall curve)
    # Negate scores because sklearn expects higher = more positive class
    ap = average_precision_score(y_true, -scores)
    
    results = {
        'model_name': model_name,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ap': ap
    }
    
    return results


def print_evaluation_report(results: dict) -> None:
    """
    Print a detailed evaluation report for a model.
    
    Parameters
    ----------
    results : dict
        Dictionary from evaluate_model()
    """
    print(f"\n{'=' * 60}")
    print(f"EVALUATION: {results['model_name']}")
    print("=" * 60)
    
    print(f"\nConfusion Matrix:")
    print(f"                        Predicted Normal    Predicted Anomaly")
    print(f"    Actual Normal       {results['tn']:>15,}    {results['fp']:>17,}")
    print(f"    Actual Fraud        {results['fn']:>15,}    {results['tp']:>17,}")
    
    print(f"\nResults:")
    print(f"    - Frauds caught: {results['tp']} out of {results['tp'] + results['fn']} "
          f"({results['recall'] * 100:.1f}%)")
    print(f"    - False alarms: {results['fp']:,} legitimate transactions flagged")
    print(f"    - Missed frauds: {results['fn']}")
    
    print(f"\nMetrics:")
    print(f"    - Precision: {results['precision']:.3f}")
    print(f"    - Recall: {results['recall']:.3f}")
    print(f"    - F1 Score: {results['f1']:.3f}")
    print(f"    - Average Precision: {results['ap']:.3f}")


def compare_models(results_list: list) -> None:
    """
    Print a comparison table of multiple models.
    
    Parameters
    ----------
    results_list : list
        List of result dictionaries from evaluate_model()
    """
    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON")
    print("=" * 70)
    
    # Header
    header = f"{'Metric':<25}"
    for r in results_list:
        header += f" {r['model_name']:>18}"
    print(header)
    print("-" * 70)
    
    # Metrics
    metrics = [
        ('Frauds Detected', 'tp', ''),
        ('False Alarms', 'fp', ''),
        ('Missed Frauds', 'fn', ''),
        ('Recall', 'recall', '%'),
        ('Precision', 'precision', '%'),
        ('F1 Score', 'f1', ''),
        ('Average Precision', 'ap', '')
    ]
    
    for name, key, fmt in metrics:
        row = f"{name:<25}"
        for r in results_list:
            if fmt == '%':
                row += f" {r[key] * 100:>17.1f}%"
            elif isinstance(r[key], int):
                row += f" {r[key]:>18,}"
            else:
                row += f" {r[key]:>18.3f}"
        print(row)
    
    print("=" * 70)


def get_precision_recall_data(y_true: np.ndarray, scores: np.ndarray) -> tuple:
    """
    Get precision-recall curve data.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    scores : np.ndarray
        Anomaly scores
    
    Returns
    -------
    tuple
        (precision, recall, thresholds)
    """
    # Negate scores because sklearn expects higher = more positive class
    precision, recall, thresholds = precision_recall_curve(y_true, -scores)
    return precision, recall, thresholds


def save_predictions(y_true: pd.Series, predictions: dict, 
                     scores: dict, filepath: str = 'predictions.csv') -> None:
    """
    Save predictions to a CSV file.
    
    Parameters
    ----------
    y_true : pd.Series
        True labels
    predictions : dict
        Dictionary of predictions for each model
    scores : dict
        Dictionary of scores for each model
    filepath : str
        Output file path
    """
    output_df = pd.DataFrame({
        'actual_class': y_true,
        'pred_isolation_forest': predictions['isolation_forest'],
        'pred_lof': predictions['lof'],
        'score_isolation_forest': scores['isolation_forest'],
        'score_lof': scores['lof']
    })
    
    output_df.to_csv(filepath, index=False)
    print(f"\n[OUTPUT] Predictions saved to: {filepath}")
