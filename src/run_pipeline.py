"""Run the end-to-end healthcare readmission modeling pipeline."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preparation import prepare_data
from src.evaluation import calculate_metrics, summarize_risk_categories
from src.feature_engineering import engineer_features
from src.modeling import (
    FEATURES,
    TARGET,
    SELECTED_THRESHOLD,
    save_model_artifacts,
    train_calibrated_random_forest,
)


RAW_DATA_PATH = Path("data/raw/diabetic_data.csv")
PREPARED_DATA_PATH = Path("data/processed/diabetic_clean_stage1.csv")
FEATURED_DATA_PATH = Path("data/processed/diabetic_features.csv")

REPORT_DIR = Path("reports")


def create_patient_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split encounters by patient so patients cannot cross train/test sets."""

    unique_patients = df["patient_nbr"].unique()

    train_patients, test_patients = train_test_split(
        unique_patients,
        test_size=0.20,
        random_state=42,
    )

    train_df = df[
        df["patient_nbr"].isin(train_patients)
    ].copy()

    test_df = df[
        df["patient_nbr"].isin(test_patients)
    ].copy()

    overlap = (
        set(train_df["patient_nbr"])
        & set(test_df["patient_nbr"])
    )

    if overlap:
        raise RuntimeError(
            f"Patient leakage detected: {len(overlap)} overlapping patients."
        )

    return train_df, test_df


def run_pipeline() -> None:
    """Run preparation, feature engineering, modeling, and reporting."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Place diabetic_data.csv in "
            "data/raw/ before running the pipeline."
        )

    print("Step 1/5: Preparing data...")
    prepare_data(
        input_path=RAW_DATA_PATH,
        output_path=PREPARED_DATA_PATH,
    )

    print("Step 2/5: Engineering features...")
    df = engineer_features(
        input_path=PREPARED_DATA_PATH,
        output_path=FEATURED_DATA_PATH,
    )

    print("Step 3/5: Creating patient-level split...")
    train_df, test_df = create_patient_split(df)

    print(
        f"Training patients: "
        f"{train_df['patient_nbr'].nunique():,}"
    )
    print(
        f"Testing patients: "
        f"{test_df['patient_nbr'].nunique():,}"
    )
    print(
        f"Training encounters: {len(train_df):,}"
    )
    print(
        f"Testing encounters: {len(test_df):,}"
    )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print("Step 4/5: Training calibrated Random Forest...")

    (
        preprocessor,
        rf_model,
        calibrated_rf,
        feature_names,
    ) = train_calibrated_random_forest(
        X_train,
        y_train,
    )

    X_test_processed = preprocessor.transform(X_test)

    metrics = calculate_metrics(
        y_true=y_test,
        probabilities=calibrated_rf.predict_proba(
            X_test_processed
        )[:, 1],
        threshold=SELECTED_THRESHOLD,
    )

    risk_summary = summarize_risk_categories(
        y_true=y_test,
        probabilities=calibrated_rf.predict_proba(
            X_test_processed
        )[:, 1],
    )

    print("Step 5/5: Saving model artifacts and reports...")

    save_model_artifacts(
        preprocessor=preprocessor,
        rf_model=rf_model,
        calibrated_rf=calibrated_rf,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_for_csv = {
        key: value
        for key, value in metrics.items()
        if key != "confusion_matrix"
    }

    pd.DataFrame(
        [metrics_for_csv]
    ).to_csv(
        REPORT_DIR / "model_performance.csv",
        index=False,
    )

    risk_summary.to_csv(
        REPORT_DIR / "risk_category_summary.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "feature": feature_names,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    ).to_csv(
        REPORT_DIR / "rf_feature_importance.csv",
        index=False,
    )

    print("\nPipeline completed successfully.")
    print("\nEvaluation metrics:")

    for key, value in metrics_for_csv.items():
        print(f"{key}: {value:.4f}")

    print(
        "\nReports saved to: "
        f"{REPORT_DIR.resolve()}"
    )

    print(
        "Model artifacts saved to: "
        f"{Path('models').resolve()}"
    )


if __name__ == "__main__":
    run_pipeline()