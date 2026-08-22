# evaluation.py

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def evaluate_model(y_true, y_pred):
    """
    Evaluate a multiclass classification model.

    Parameters
    ----------
    y_true : array-like
        True class labels.

    y_pred : array-like
        Predicted class labels.

    Returns
    -------
    metrics : dict
        Dictionary containing accuracy, precision,
        recall, and F1 scores.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "f1_weighted": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }

    return metrics

def get_classification_report(y_true, y_pred):
    """
    Generate a classification report as a DataFrame.
    """

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    return pd.DataFrame(report).transpose()

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    title="Confusion Matrix"
):
    """
    Plot a confusion matrix.
    """

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        values_format="d"
    )

    plt.title(title)
    plt.tight_layout()
    plt.show()

    return cm

def compare_models(predictions, y_true):
    """
    Compare multiple models using the same test set.

    Parameters
    ----------
    predictions : dict
        Dictionary where keys are model names and values
        are predicted labels.

    y_true : array-like
        True test labels.

    Returns
    -------
    results : pandas.DataFrame
        Model comparison table.
    """

    results = []

    for model_name, y_pred in predictions.items():

        metrics = evaluate_model(
            y_true,
            y_pred
        )

        metrics["model"] = model_name

        results.append(metrics)

    results = pd.DataFrame(results)

    results = results[
        [
            "model",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted"
        ]
    ]

    results = results.sort_values(
        by="f1_macro",
        ascending=False
    ).reset_index(drop=True)

    return results

def save_model_comparison(
    results,
    filepath="results/model_comparison.csv"
):
    """
    Save model comparison results to CSV.
    """

    results.to_csv(
        filepath,
        index=False
    )

    return filepath

def get_cv_result(search):
    """
    Extract the best cross-validation score
    and parameters from a fitted search object.
    """

    return {
        "best_cv_f1_macro": search.best_score_,
        "best_params": search.best_params_
    }