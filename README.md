# Predicting-Anaemia-Severity-Among-Pregnant-Women-using-NDHS
Machine learning and deep learning models for predicting anaemia severity among pregnant Nigerian women using the 2024 NDHS — with SHAP interpretability analysis.

# Predicting Anaemia Severity Among Pregnant Women in Nigeria
### A Machine Learning and Deep Learning Study Using the 2024 NDHS

## Overview
This repository contains the full source code for my dissertation project, which 
develops and evaluates machine learning and deep learning models for predicting 
anaemia severity among pregnant women in Nigeria, using data from the 2024 Nigeria 
Demographic and Health Survey (NDHS).

## Models Implemented
- Logistic Regression (with SMOTE, hyperparameter tuning, threshold adjustment)
- Random Forest (with SMOTE, hyperparameter tuning, threshold adjustment)
- XGBoost (with sample weighting, hyperparameter tuning, threshold adjustment)
- Artificial Neural Network (ANN) (with SMOTE)

## Key Findings
- Logistic Regression with SMOTE achieved the best macro-average F1-score (0.350)
- SHAP analysis identified region, age group, and religion as the three dominant predictors
- Ensemble method dominance over linear classifiers was reversed under small sample conditions

## Tools and Libraries
- Python 3.12 | Google Colaboratory
- scikit-learn, XGBoost, TensorFlow/Keras
- SHAP, imbalanced-learn, pandas, matplotlib, seaborn

## Repository Contents
- `anaemia_severity_prediction.ipynb` — full annotated source code including
  preprocessing, model training, evaluation, and SHAP interpretability analysis

## Data
The dataset is sourced from the 2024 Nigeria Demographic and Health Survey (NDHS).
Raw data is not included in this repository in compliance with DHS data use agreements.
To access the data, apply at: https://dhsprogram.com/data/available-datasets.cfm

## Author
ADEGBOYE, ABISADE MARY | REDEEMER'S UNIVERSITY | 2026
