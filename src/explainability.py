# explainability.py

import numpy as np
import pandas as pd
import shap

from scipy import sparse

def get_feature_names(
    preprocessor,
    numeric_cols,
    low_card_cat_cols,
    high_card_cat_cols,
    ordinal_cols
):
    """
    Construct feature names after preprocessing.

    CatBoostEncoder produces one column per original
    high-cardinality categorical variable, so those
    names are added manually.
    """

    feature_names = []

    # Numeric features
    feature_names.extend(numeric_cols)

    # One-hot encoded features
    ohe = (
        preprocessor
        .named_transformers_["low_cat"]
        .get_feature_names_out(low_card_cat_cols)
        .tolist()
    )

    feature_names.extend(ohe)

    # CatBoost-encoded features
    feature_names.extend(high_card_cat_cols)

    # Ordinal features
    feature_names.extend(ordinal_cols)

    return feature_names

def transform_for_explanation(
    preprocessor,
    X
):
    """
    Transform data using an already-fitted preprocessor
    and convert sparse output to dense format when necessary.
    """

    X_transformed = preprocessor.transform(X)

    if sparse.issparse(X_transformed):
        X_transformed = X_transformed.toarray()

    return np.asarray(X_transformed)

def calculate_tree_shap(
    model,
    preprocessor,
    X_background,
    X_explain
):
    """
    Calculate SHAP values for a fitted tree-based model.

    Parameters
    ----------
    model : fitted tree model
        For example, RandomForestClassifier.

    preprocessor : fitted transformer
        Preprocessing pipeline used by the model.

    X_background : DataFrame
        Background data used by SHAP.

    X_explain : DataFrame
        Data for which SHAP values are calculated.

    Returns
    -------
    shap_values : numpy.ndarray
        SHAP values.
    """

    X_background_transformed = transform_for_explanation(
        preprocessor,
        X_background
    )

    X_explain_transformed = transform_for_explanation(
        preprocessor,
        X_explain
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_explain_transformed
    )

    return shap_values

def calculate_shap_importance(
    shap_values,
    feature_names
):
    """
    Calculate mean absolute SHAP importance
    for each feature.
    """

    shap_array = np.asarray(shap_values)

    # Multiclass SHAP:
    # (n_samples, n_features, n_classes)
    if shap_array.ndim == 3:

        importance = np.abs(
            shap_array
        ).mean(axis=(0, 2))

    # Binary/single-output SHAP:
    elif shap_array.ndim == 2:

        importance = np.abs(
            shap_array
        ).mean(axis=0)

    else:
        raise ValueError(
            f"Unexpected SHAP shape: {shap_array.shape}"
        )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_SHAP": importance
    })

    importance_df = (
        importance_df
        .sort_values(
            "Mean_SHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return importance_df

def aggregate_shap_importance(
    shap_importance,
    feature_groups
):
    """
    Aggregate transformed feature SHAP values
    back to their original variables.

    Parameters
    ----------
    shap_importance : DataFrame
        Output from calculate_shap_importance().

    feature_groups : dict
        Maps original variables to their transformed
        feature names.
    """

    aggregated = []

    for original_variable, transformed_features in feature_groups.items():

        total = (
            shap_importance[
                shap_importance["Feature"].isin(
                    transformed_features
                )
            ]["Mean_SHAP"]
            .sum()
        )

        aggregated.append({
            "Feature": original_variable,
            "Mean_SHAP": total
        })

    aggregated_df = (
        pd.DataFrame(aggregated)
        .sort_values(
            "Mean_SHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )

    total_importance = aggregated_df["Mean_SHAP"].sum()

    aggregated_df["% Contribution"] = (
        aggregated_df["Mean_SHAP"]
        / total_importance
        * 100
    ).round(1)

    return aggregated_df

def get_tree_feature_importance(
    model,
    feature_names
):
    """
    Extract native feature importance from a
    fitted tree-based model.
    """

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            "The supplied model does not have "
            "'feature_importances_'."
        )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    return (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )