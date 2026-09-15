"""Train and evaluate the calibrated Random Forest readmission model."""

from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
SELECTED_THRESHOLD = 0.14

NUMERIC_FEATURES: List[str] = [
    "age_numeric",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "total_prior_visits",
    "prior_inpatient_visit",
    "diabetes_diagnosis",
    "diagnosis_category_count",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "race",
    "admission_type",
    "admission_source",
    "diag_1_category",
    "diag_2_category",
    "diag_3_category",
    "payer_code",
    "medical_specialty",
]

FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "readmitted_30"

MODEL_DIR = Path("models")

def create_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline used by the Random Forest."""

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

def train_calibrated_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[ColumnTransformer, RandomForestClassifier, CalibratedClassifierCV, object]:
    """Fit the Random Forest and its calibrated probability model."""

    preprocessor = create_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf_model.fit(X_train_processed, y_train)

    calibrated_rf = CalibratedClassifierCV(
        estimator=rf_model,
        method="sigmoid",
        cv=3,
    )

    calibrated_rf.fit(X_train_processed, y_train)

    feature_names = preprocessor.get_feature_names_out()

    return (
        preprocessor,
        rf_model,
        calibrated_rf,
        feature_names,
    )

def evaluate_model(
    model,
    X_test_processed,
    y_test: pd.Series,
    threshold: float = 0.14,
) -> Dict[str, object]:
    """Evaluate model probabilities at the selected decision threshold."""

    predicted_probability = model.predict_proba(X_test_processed)[:, 1]

    predicted_class = (
        predicted_probability >= threshold
    ).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predicted_class),
        "precision": precision_score(
            y_test,
            predicted_class,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predicted_class,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predicted_class,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            predicted_probability,
        ),
        "pr_auc": average_precision_score(
            y_test,
            predicted_probability,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predicted_class,
        ),
    }

    return metrics

def save_model_artifacts(
    preprocessor: ColumnTransformer,
    rf_model: RandomForestClassifier,
    calibrated_rf: CalibratedClassifierCV,
) -> None:
    """Save preprocessing and trained model artifacts."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        preprocessor,
        MODEL_DIR / "rf_preprocessor.joblib",
    )

    joblib.dump(
        rf_model,
        MODEL_DIR / "random_forest.joblib",
    )

    joblib.dump(
        calibrated_rf,
        MODEL_DIR / "calibrated_random_forest.joblib",
    )