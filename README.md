# Credit Card Fraud Detection System

Detecting fraudulent credit card transactions using Machine Learning.

## Problem Statement
Credit card fraud is a major financial threat. This project builds an ML 
model to identify fraudulent transactions from a highly imbalanced dataset 
where only 0.17% of transactions are fraud.

## Dataset
Kaggle Credit Card Fraud Detection Dataset  
- 284,807 transactions  
- 492 fraud cases (0.17% fraud rate)  
- Features: V1-V28 (PCA transformed), Amount, Time

## What I Did
- Feature Engineering: log-transform on Amount to reduce skewness, 
  extracted hour-of-day from Time column
- Built and compared 3 models: Logistic Regression, Decision Tree, Random Forest
- Evaluated using ROC-AUC, Precision, Recall, and Confusion Matrix
- Identified top 10 most important features using Random Forest feature importances
- Prioritized Recall over Accuracy to minimize undetected fraud cases

## Libraries Used
- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## How to Run
1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place `creditcard.csv` in the same folder as the notebook
3. Open `Project_code.ipynb` in Jupyter Notebook or Google Colab
4. Run all cells

## Results
- Random Forest achieved the best performance among all 3 models
- Evaluated on ROC-AUC score and Recall metric# credit-card-fraud-detection
ML model to detect fraudulent transactions using Logistic Regression, Decision Tree  and Random Forest
