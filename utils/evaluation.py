
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate(y_true, y_pred):
    """
    Calculate classification metrics.

    Returns a dictionary containing accuracy and macro-F1.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Prediction and label shapes must match.")

    if y_true.ndim != 1:
        raise ValueError("Labels must be one-dimensional.")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
    }


def print_report(y_true, y_pred):
    """Print precision, recall and F1-score for each class."""

    print(classification_report(
        y_true,
        y_pred,
        zero_division=0
    ))


def get_confusion_matrix(y_true, y_pred, labels=None):
    """Return the confusion matrix."""

    return confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Evaluate a fitted model on training and validation data.

    Useful for detecting overfitting and underfitting.
    """

    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

    train_metrics = evaluate(y_train, train_predictions)
    val_metrics = evaluate(y_val, val_predictions)

    return {
        "train_accuracy": train_metrics["accuracy"],
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
    }
