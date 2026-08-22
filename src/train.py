# train.py
# src/train.py

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

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

def create_cv():
    """
    Create a stratified cross-validation splitter
    for multiclass classification.
    """

    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
def tune_logistic_regression(X_train, y_train):
    """
    Tune Logistic Regression hyperparameters using
    stratified cross-validation and macro F1.
    """

    pipeline = build_logistic_pipeline()

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10]
    }

    cv = create_cv()

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    return search

def build_logistic_smote_pipeline():

    pipeline = ImbPipeline([
        (
            "outlier_capper",
            IQRCapper(columns=outlier_cols)
        ),

        (
            "preprocess",
            create_scaled_preprocessor()
        ),

        (
            "smote",
            SMOTE(random_state=42)
        ),

        (
            "model",
            LogisticRegression(
                solver="lbfgs",
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        )
    ])

    return pipeline

def tune_logistic_smote(X_train, y_train):
    """
    Tune a SMOTE Logistic Regression pipeline
    using stratified cross-validation and macro F1.
    """

    pipeline = build_logistic_smote_pipeline()

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "smote__k_neighbors": [3, 5, 7]
    }

    cv = create_cv()

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    return search

