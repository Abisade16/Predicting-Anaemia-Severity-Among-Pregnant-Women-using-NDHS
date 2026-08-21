# src/preprocessing.py

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from category_encoders import CatBoostEncoder


# ============================================================
# 1. COLUMN GROUPS
# ============================================================

numeric_cols = [
    "BMI",
    "total_children_ever_born",
    "births_in_the_last_5_years",
    "antenatal_visits"
]

low_card_cat_cols = [
    "age_group",
    "region",
    "type_of_residence",
    "religion",
    "had_STI_last_12months",
    "currently_breastfeeding",
    "marital_status",
    "currently_working"
]

high_card_cat_col = [
    "ethnicity"
]

ordinal_cols = [
    "birth_interval",
    "education_level",
    "wealth_index",
    "freq_of_reading_newspapers",
    "freq_of_listening_to_radio",
    "freq_of_watching_TV"
]


# ============================================================
# 2. OUTLIER COLUMNS
# ============================================================

outlier_cols = [
    "BMI",
    "births_in_the_last_5_years",
    "total_children_ever_born"
]


# ============================================================
# 3. ORDINAL MAPPINGS
# ============================================================

birth_interval_order = {
    "No previous birth": 0,
    "Short interval": 1,
    "Medium interval": 2,
    "Long interval": 3
}

education_level_order = {
    "No education": 0,
    "Primary": 1,
    "Secondary": 2,
    "Higher": 3
}

wealth_index_order = {
    "Poorest": 0,
    "Poorer": 1,
    "Middle": 2,
    "Richer": 3,
    "Richest": 4
}

media_order = {
    "Not at all": 0,
    "Less than once a week": 1,
    "At least once a week": 2
}


# ============================================================
# 4. ORDINAL ENCODING
# ============================================================

def encode_ordinals(df):
    """
    Convert ordinal categorical variables into numerical
    values while preserving their natural order.
    """

    df = df.copy()

    df["birth_interval"] = (
        df["birth_interval"].map(birth_interval_order)
    )

    df["education_level"] = (
        df["education_level"].map(education_level_order)
    )

    df["wealth_index"] = (
        df["wealth_index"].map(wealth_index_order)
    )

    df["freq_of_reading_newspapers"] = (
        df["freq_of_reading_newspapers"].map(media_order)
    )

    df["freq_of_listening_to_radio"] = (
        df["freq_of_listening_to_radio"].map(media_order)
    )

    df["freq_of_watching_TV"] = (
        df["freq_of_watching_TV"].map(media_order)
    )

    return df


# ============================================================
# 5. BMI CONVERSION
# ============================================================

def convert_bmi(df):
    """
    Convert BMI from the NDHS stored representation
    to the actual BMI value.
    """

    df = df.copy()
    df["BMI"] = df["BMI"] / 100

    return df


# ============================================================
# 6. IQR OUTLIER CAPPING
# ============================================================

class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Cap outliers using the Interquartile Range (IQR) method.

    The IQR boundaries are learned during fit() and applied
    during transform().

    Because this transformer is placed inside an sklearn
    Pipeline, the boundaries are learned only from the
    training portion of each cross-validation fold.
    """

    def __init__(self, columns, multiplier=1.5):
        self.columns = columns
        self.multiplier = multiplier

    def fit(self, X, y=None):

        X = X.copy()

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for col in self.columns:

            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)

            IQR = Q3 - Q1

            self.lower_bounds_[col] = (
                Q1 - self.multiplier * IQR
            )

            self.upper_bounds_[col] = (
                Q3 + self.multiplier * IQR
            )

        return self

    def transform(self, X):

        X = X.copy()

        for col in self.columns:

            X[col] = X[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )

        return X


# ============================================================
# 7. LOGISTIC REGRESSION / ANN PREPROCESSOR
# ============================================================

def create_scaled_preprocessor():

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "num_scale",
                StandardScaler(),
                numeric_cols
            ),

            (
                "low_cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first"
                ),
                low_card_cat_cols
            ),

            (
                "high_cat",
                CatBoostEncoder(),
                high_card_cat_col
            ),

            (
                "ord",
                StandardScaler(),
                ordinal_cols
            )
        ]
    )

    return preprocessor


# ============================================================
# 8. RANDOM FOREST / XGBOOST PREPROCESSOR
# ============================================================

def create_tree_preprocessor():

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "num_pass",
                "passthrough",
                numeric_cols
            ),

            (
                "low_cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                low_card_cat_cols
            ),

            (
                "high_cat",
                CatBoostEncoder(),
                high_card_cat_col
            ),

            (
                "ord",
                "passthrough",
                ordinal_cols
            )
        ]
    )

    return preprocessor