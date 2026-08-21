# train.py
# src/train.py

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.preprocessing import (
    IQRCapper,
    create_scaled_preprocessor,
    create_tree_preprocessor,
    outlier_cols
)


# ============================================================
# 1. LOGISTIC REGRESSION
# ============================================================

def build_logistic_pipeline():

    pipeline = Pipeline([
        (
            "outlier_capper",
            IQRCapper(columns=outlier_cols)
        ),

        (
            "preprocess",
            create_scaled_preprocessor()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    return pipeline


# ============================================================
# 2. RANDOM FOREST
# ============================================================

def build_random_forest_pipeline():

    pipeline = Pipeline([
        (
            "outlier_capper",
            IQRCapper(columns=outlier_cols)
        ),

        (
            "preprocess",
            create_tree_preprocessor()
        ),

        (
            "model",
            RandomForestClassifier(
                random_state=42
            )
        )
    ])

    return pipeline


# ============================================================
# 3. XGBOOST
# ============================================================

def build_xgboost_pipeline():

    pipeline = Pipeline([
        (
            "outlier_capper",
            IQRCapper(columns=outlier_cols)
        ),

        (
            "preprocess",
            create_tree_preprocessor()
        ),

        (
            "model",
            XGBClassifier(
                random_state=42,
                eval_metric="mlogloss"
            )
        )
    ])

    return pipeline