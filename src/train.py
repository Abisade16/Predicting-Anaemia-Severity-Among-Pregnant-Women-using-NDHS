# train.py
# src/train.py
from scipy.stats import randint, uniform
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.preprocessing import (
    IQRCapper,
    create_scaled_preprocessor,
    create_tree_preprocessor,
    outlier_cols
)

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



def tune_random_forest(X_train, y_train):
    """
    Tune Random Forest hyperparameters using
    RandomizedSearchCV and macro F1.
    """

    pipeline = build_random_forest_pipeline()

    param_dist = {
        "model__max_depth": [
            None,
            10,
            20,
            30
        ],

        "model__min_samples_split": [
            2,
            5,
            10
        ],

        "model__min_samples_leaf": [
            1,
            2,
            4
        ],

        "model__max_features": [
            "sqrt",
            "log2",
            None
        ]
    }

    cv = create_cv()

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    return search

def build_random_forest_smote_pipeline():

    pipeline = ImbPipeline([
        (
            "outlier_capper",
            IQRCapper(columns=outlier_cols)
        ),

        (
            "preprocess",
            create_tree_preprocessor()
        ),

        (
            "smote",
            SMOTE(random_state=42)
        ),

        (
            "model",
            RandomForestClassifier(
                random_state=42
            )
        )
    ])

    return pipeline

def tune_random_forest_smote(X_train, y_train):
    """
    Tune a SMOTE Random Forest pipeline using
    RandomizedSearchCV and macro F1.
    """

    pipeline = build_random_forest_smote_pipeline()

    param_dist = {
        "model__max_depth": [
            None,
            10,
            20,
            30
        ],

        "model__min_samples_split": [
            2,
            5,
            10
        ],

        "model__min_samples_leaf": [
            1,
            2,
            4
        ],

        "model__max_features": [
            "sqrt",
            "log2",
            None
        ],

        "smote__k_neighbors": [
            3,
            5,
            7
        ]
    }

    cv = create_cv()

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=50,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    return search

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
                objective="multi:softmax",
                num_class=4,
                eval_metric="mlogloss",
                random_state=42
            )
        )
    ])

    return pipeline

def tune_xgboost(X_train, y_train):
    """
    Tune XGBoost hyperparameters using RandomizedSearchCV
    and macro F1.
    """

    pipeline = build_xgboost_pipeline()

    param_dist = {
        "model__n_estimators": randint(100, 500),

        "model__max_depth": randint(3, 10),

        "model__learning_rate": uniform(0.01, 0.3),

        "model__subsample": uniform(0.7, 0.3),

        "model__colsample_bytree": uniform(0.7, 0.3),

        "model__gamma": uniform(0, 5)
    }

    cv = create_cv()

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=50,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    return search

def train_xgboost_with_sample_weights(X_train, y_train):
    """
    Train XGBoost using balanced sample weights
    to account for class imbalance.

    Returns
    -------
    pipeline : fitted Pipeline
        The fitted XGBoost pipeline.
    """

    pipeline = build_xgboost_pipeline()

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    pipeline.fit(
        X_train,
        y_train,
        model__sample_weight=sample_weights
    )

    return pipeline