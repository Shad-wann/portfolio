# Credit Card Fraud Detection

> Unsupervised Anomaly Detection for Financial Transactions

# Overview

This project explores **unsupervised machine learning** techniques to detect fraudulent credit card transactions. The core idea is simple: frauds are rare and unusual, so they should stand out as anomalies in the data.

# Key Features
- **Anomaly Detection** using Isolation Forest and Local Outlier Factor (LOF)
- **Exploratory Data Analysis** with comprehensive visualizations
- **Model Evaluation** with precision-recall metrics adapted for imbalanced data
- **Clean, modular code** suitable for a portfolio project

# Dataset

The dataset contains **284,807 transactions** made by European cardholders in September 2013. Only **492 transactions (0.17%)** are fraudulent — a classic needle-in-a-haystack problem.

| Feature | Description |
|---------|-------------|
| V1-V28 | PCA-transformed features (anonymized) |
| Time   | Seconds elapsed since first transaction |
| Amount | Transaction amount (€) |
| Class | Target: 0 = Legitimate, 1 = Fraud |

**Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

# Quick Start

# 1. Clone the repository
```bash
git clone git@github.com:Shad-wann/portfolio.git
cd credit-card-fraud-detection
```

# 2. Install dependencies
```bash
pip install -r requirements.txt
```

# 3. Download the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder.

# 4. Run the notebook
```bash
jupyter notebook notebooks/ccfdd.ipynb
```

# Project Structure

```
credit-card-fraud-detection/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── notebooks/
│   └── ccfdd.ipynb          # Main analysis notebook
├── src/
│   ├── __init__.py          # Package initialization
│   ├── data_loading.py      # Data loading utilities
│   ├── preprocessing.py     # Data preprocessing functions
│   ├── models.py            # Anomaly detection models
│   ├── evaluation.py        # Model evaluation metrics
│   └── visualization.py     # Plotting functions
├── data/
│   └── README.md            # Dataset instructions
├── outputs/
│   └── figures/             # Generated visualizations
└── reports/
    └── project_summary.md   # Project summary report
```

# Methodology

# 1. Exploratory Data Analysis
- Class distribution analysis (extreme imbalance: 99.83% vs 0.17%)
- Transaction amount patterns
- Temporal patterns
- Feature correlations

# 2. Preprocessing
- Standardization of `Time` and `Amount` features
- PCA projection for 2D visualization

# 3. Anomaly Detection Models

| Model | Principle |
|-------|-----------|
| **Isolation Forest** | Isolates anomalies using random partitioning. Anomalies require fewer splits to be isolated. |
| **Local Outlier Factor** | Compares local density of each point to its neighbors. Outliers have lower density. |

# 4. Evaluation
- Confusion Matrix
- Precision, Recall, F1-Score
- Average Precision (PR-AUC)
- Precision-Recall Curves

# Results

| Metric | Isolation Forest | LOF |
|--------|------------------|-----|
| Frauds Detected | ~350 | ~350 |
| Recall | ~70%   | ~70% |
| Precision | ~5-10% | ~5-10% |
| Average Precision | ~0.3 | ~0.3 |

> **Note:** Results may vary slightly due to random initialization.

# Sample Visualizations

The project generates several visualizations:
- `01_eda_overview.png` - Exploratory data analysis
- `02_pca_projection.png` - 2D projection of transactions
- `03_score_distributions.png` - Anomaly score distributions
- `04_precision_recall.png` - Precision-recall curves
- `05_detection_comparison.png` - Actual vs detected anomalies
- `06_results_summary.png` - Performance comparison

# Author

**Farès HAMDI**  
Specialization: Data Science & Machine Learning

