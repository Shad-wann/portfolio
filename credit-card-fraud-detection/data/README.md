# Dataset Instructions

# Credit Card Fraud Detection Dataset

This project uses the **Credit Card Fraud Detection** dataset from Kaggle.

# Download Instructions

1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

2. Download the `creditcard.csv` file (143.84 MB)

3. Place the file in this `data/` folder

### Dataset Information

| Property | Value |
|----------|-------|
| **File name** | `creditcard.csv` |
| **Size** | ~143 MB |
| **Rows** | 284,807 transactions |
| **Columns** | 31 features |
| **Time period** | 2 days in September 2013 |
| **Location** | European cardholders |

# Features Description

| Feature | Description |
|---------|-------------|
| `Time` | Seconds elapsed since the first transaction |
| `V1-V28` | Principal components from PCA (anonymized) |
| `Amount` | Transaction amount in euros (€) |
| `Class` | Target variable: 0 = Legitimate, 1 = Fraud |

# Class Distribution

- **Legitimate transactions:** 284,315 (99.83%)
- **Fraudulent transactions:** 492 (0.17%)
