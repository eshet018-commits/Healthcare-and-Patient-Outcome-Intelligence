"""Evaluate readmission models and decision thresholds."""

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.14,
) -> Dict[str, object]:
    """Calculate classification and probability-based metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
        ),
    }


def evaluate_thresholds(
    y_true: pd.Series,
    probabilities: np.ndarray,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Evaluate precision, recall, and F1 across thresholds."""

    results = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(results)


def summarize_risk_categories(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Summarize observed outcomes across predicted risk categories."""

    results = pd.DataFrame(
        {
            "actual_readmitted_30": y_true.to_numpy(),
            "predicted_risk": probabilities,
        }
    )

    results["risk_category"] = pd.cut(
        results["predicted_risk"],
        bins=[
            -np.inf,
            0.10,
            0.14,
            0.20,
            np.inf,
        ],
        labels=[
            "Lower Risk",
            "Moderate Risk",
            "High Risk",
            "Very High Risk",
        ],
        right=False,
    )

    summary = (
        results.groupby(
            "risk_category",
            observed=False,
        )
        .agg(
            encounters=(
                "actual_readmitted_30",
                "size",
            ),
            actual_readmissions=(
                "actual_readmitted_30",
                "sum",
            ),
            actual_readmission_rate=(
                "actual_readmitted_30",
                "mean",
            ),
            average_predicted_risk=(
                "predicted_risk",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary