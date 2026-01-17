"""
Credit Card Fraud Detection - Source Package

This package contains modules for fraud detection using anomaly detection algorithms.

Modules:
    - data_loading: Functions for loading and exploring the dataset
    - preprocessing: Data preprocessing and feature scaling
    - models: Anomaly detection models (Isolation Forest, LOF)
    - evaluation: Model evaluation metrics and utilities
    - visualization: Plotting functions for analysis and results
"""

from . import data_loading
from . import preprocessing
from . import models
from . import evaluation
from . import visualization

__version__ = "2.0.0"
__author__ = "Farès HAMDI"
