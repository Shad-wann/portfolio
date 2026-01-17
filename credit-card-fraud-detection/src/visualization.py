"""
Visualization Module

Functions for creating plots and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Default plot settings
def setup_plot_style():
    """Configure matplotlib style for consistent plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10


def plot_eda_overview(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Create exploratory data analysis overview plots.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    save_path : str, optional
        Path to save the figure
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Exploratory Data Analysis', fontsize=14, fontweight='bold')
    
    # 1. Class distribution (log scale)
    ax1 = axes[0, 0]
    counts = df['Class'].value_counts()
    bars = ax1.bar(['Legitimate', 'Fraud'], counts.values, 
                   color=['#27ae60', '#c0392b'], edgecolor='black')
    ax1.set_ylabel('Number of transactions (log scale)')
    ax1.set_title('Class Distribution')
    ax1.set_yscale('log')
    for bar, count in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                 f'{count:,}', ha='center', va='bottom', fontsize=10)
    
    # 2. Amount distribution
    ax2 = axes[0, 1]
    ax2.hist(df['Amount'], bins=100, color='#3498db', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Amount (€)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Transaction Amount Distribution')
    ax2.set_xlim(0, df['Amount'].quantile(0.99))
    
    # 3. Time distribution
    ax3 = axes[1, 0]
    hours = df['Time'] / 3600
    ax3.hist(hours, bins=48, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Time (hours since first transaction)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Transactions Over Time')
    
    # 4. Amount by class
    ax4 = axes[1, 1]
    legit_amounts = df[df['Class'] == 0]['Amount']
    fraud_amounts = df[df['Class'] == 1]['Amount']
    bp = ax4.boxplot([legit_amounts, fraud_amounts], 
                     labels=['Legitimate', 'Fraud'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#27ae60')
    bp['boxes'][1].set_facecolor('#c0392b')
    ax4.set_ylabel('Amount (€)')
    ax4.set_title('Amount by Class')
    ax4.set_ylim(0, 500)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VIZ] Saved: {save_path}")
    
    plt.show()


def plot_pca_projection(X_2d: np.ndarray, y: pd.Series, pca, 
                        save_path: str = None) -> None:
    """
    Plot 2D PCA projection of transactions.
    
    Parameters
    ----------
    X_2d : np.ndarray
        2D projected features
    y : pd.Series
        Target variable
    pca : PCA
        Fitted PCA object
    save_path : str, optional
        Path to save the figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sample for readability
    np.random.seed(42)
    sample_legit = np.random.choice(np.where(y == 0)[0], size=10000, replace=False)
    sample_fraud = np.where(y == 1)[0]
    
    ax.scatter(X_2d[sample_legit, 0], X_2d[sample_legit, 1],
               c='#27ae60', alpha=0.3, s=10, label=f'Legitimate (n=10,000)')
    ax.scatter(X_2d[sample_fraud, 0], X_2d[sample_fraud, 1],
               c='#c0392b', alpha=0.8, s=30, label=f'Fraud (n={len(sample_fraud)})')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
    ax.set_title('2D Projection of Transactions')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VIZ] Saved: {save_path}")
    
    plt.show()


def plot_score_distributions(y: pd.Series, scores_iso: np.ndarray, 
                              scores_lof: np.ndarray, save_path: str = None) -> None:
    """
    Plot anomaly score distributions for both models.
    
    Parameters
    ----------
    y : pd.Series
        True labels
    scores_iso : np.ndarray
        Isolation Forest scores
    scores_lof : np.ndarray
        LOF scores
    save_path : str, optional
        Path to save the figure
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Isolation Forest
    ax1 = axes[0]
    ax1.hist(scores_iso[y == 0], bins=100, alpha=0.7, 
             label='Legitimate', color='#27ae60', density=True)
    ax1.hist(scores_iso[y == 1], bins=50, alpha=0.7, 
             label='Fraud', color='#c0392b', density=True)
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Isolation Forest - Score Distribution')
    ax1.legend()
    
    # LOF
    ax2 = axes[1]
    ax2.hist(scores_lof[y == 0], bins=100, alpha=0.7, 
             label='Legitimate', color='#27ae60', density=True)
    ax2.hist(scores_lof[y == 1], bins=50, alpha=0.7, 
             label='Fraud', color='#c0392b', density=True)
    ax2.set_xlabel('Anomaly Score')
    ax2.set_ylabel('Density')
    ax2.set_title('LOF - Score Distribution')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VIZ] Saved: {save_path}")
    
    plt.show()


def plot_precision_recall_curves(y: pd.Series, scores_iso: np.ndarray, 
                                  scores_lof: np.ndarray, results_iso: dict,
                                  results_lof: dict, save_path: str = None) -> None:
    """
    Plot precision-recall curves for both models.
    
    Parameters
    ----------
    y : pd.Series
        True labels
    scores_iso : np.ndarray
        Isolation Forest scores
    scores_lof : np.ndarray
        LOF scores
    results_iso : dict
        Evaluation results for Isolation Forest
    results_lof : dict
        Evaluation results for LOF
    save_path : str, optional
        Path to save the figure
    """
    from sklearn.metrics import precision_recall_curve
    
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute curves
    prec_iso, rec_iso, _ = precision_recall_curve(y, -scores_iso)
    prec_lof, rec_lof, _ = precision_recall_curve(y, -scores_lof)
    
    ax.plot(rec_iso, prec_iso, 
            label=f'Isolation Forest (AP={results_iso["ap"]:.3f})',
            color='#3498db', linewidth=2)
    ax.plot(rec_lof, prec_lof, 
            label=f'LOF (AP={results_lof["ap"]:.3f})',
            color='#e67e22', linewidth=2)
    
    # Random baseline
    baseline = y.sum() / len(y)
    ax.axhline(y=baseline, color='gray', linestyle='--', 
               label=f'Random baseline ({baseline:.4f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='upper right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VIZ] Saved: {save_path}")
    
    plt.show()


def plot_detection_comparison(X_2d: np.ndarray, y: pd.Series, 
                               pred_iso: np.ndarray, pred_lof: np.ndarray,
                               save_path: str = None) -> None:
    """
    Plot comparison of actual frauds vs detected anomalies in 2D.
    
    Parameters
    ----------
    X_2d : np.ndarray
        2D projected features
    y : pd.Series
        True labels
    pred_iso : np.ndarray
        Isolation Forest predictions
    pred_lof : np.ndarray
        LOF predictions
    save_path : str, optional
        Path to save the figure
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Sample for readability
    np.random.seed(42)
    sample = np.random.choice(len(y), size=12000, replace=False)
    
    # Actual frauds
    ax1 = axes[0]
    colors = ['#27ae60' if label == 0 else '#c0392b' for label in y.iloc[sample]]
    ax1.scatter(X_2d[sample, 0], X_2d[sample, 1], c=colors, alpha=0.4, s=8)
    ax1.set_title('Actual Frauds')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    
    # Isolation Forest
    ax2 = axes[1]
    colors = ['#27ae60' if pred == 0 else '#c0392b' for pred in pred_iso[sample]]
    ax2.scatter(X_2d[sample, 0], X_2d[sample, 1], c=colors, alpha=0.4, s=8)
    ax2.set_title('Isolation Forest Detections')
    ax2.set_xlabel('PC1')
    
    # LOF
    ax3 = axes[2]
    colors = ['#27ae60' if pred == 0 else '#c0392b' for pred in pred_lof[sample]]
    ax3.scatter(X_2d[sample, 0], X_2d[sample, 1], c=colors, alpha=0.4, s=8)
    ax3.set_title('LOF Detections')
    ax3.set_xlabel('PC1')
    
    plt.suptitle('Comparison: Actual vs Detected (Green = Normal, Red = Anomaly)', 
                 fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VIZ] Saved: {save_path}")
    
    plt.show()


def plot_results_summary(results_iso: dict, results_lof: dict, 
                          save_path: str = None) -> None:
    """
    Plot summary bar charts of model results.
    
    Parameters
    ----------
    results_iso : dict
        Evaluation results for Isolation Forest
    results_lof : dict
        Evaluation results for LOF
    save_path : str, optional
        Path to save the figure
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    width = 0.35
    
    # Detection counts
    ax1 = axes[0]
    categories = ['True Positives\n(Caught)', 'False Positives\n(False Alarms)', 
                  'False Negatives\n(Missed)']
    x = np.arange(len(categories))
    
    ax1.bar(x - width/2, [results_iso['tp'], results_iso['fp'], results_iso['fn']],
            width, label='Isolation Forest', color='#3498db')
    ax1.bar(x + width/2, [results_lof['tp'], results_lof['fp'], results_lof['fn']],
            width, label='LOF', color='#e67e22')
    
    ax1.set_ylabel('Count (log scale)')
    ax1.set_title('Detection Results')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.set_yscale('log')
    
    # Performance metrics
    ax2 = axes[1]
    metrics = ['Recall', 'Precision']
    x = np.arange(len(metrics))
    
    bars1 = ax2.bar(x - width/2, 
                    [results_iso['recall'] * 100, results_iso['precision'] * 100],
                    width, label='Isolation Forest', color='#3498db')
    bars2 = ax2.bar(x + width/2, 
                    [results_lof['recall'] * 100, results_lof['precision'] * 100],
                    width, label='LOF', color='#e67e22')
    
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Performance Metrics')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.set_ylim(0, 100)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VIZ] Saved: {save_path}")
    
    plt.show()
