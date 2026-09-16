"""Load reproducible healthcare analytics outputs into PostgreSQL."""

from pathlib import Path

import pandas as pd

from src.database import get_engine, load_dataframe_to_table


FEATURES_PATH = Path("data/processed/diabetic_features.csv")
PREDICTIONS_PATH = Path(
    "data/processed/readmission_risk_predictions.csv"
)


ENCOUNTER_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "age",
    "age_numeric",
    "gender",
    "race",
    "admission_type_id",
    "admission_type",
    "admission_source_id",
    "admission_source",
    "discharge_disposition_id",
    "discharge_disposition",
    "time_in_hospital",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "total_prior_visits",
    "prior_inpatient_visit",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "medication_count",
    "medication_changed",
    "diabetes_medication",
    "diabetes_diagnosis",
    "diagnosis_category_count",
    "diag_1_category",
    "diag_2_category",
    "diag_3_category",
    "high_medication_burden",
    "long_stay",
    "payer_code",
    "medical_specialty",
    "readmitted",
    "readmitted_30",
]


PREDICTION_COLUMNS = [
    "encounter_id",
    "predicted_readmission_risk",
    "risk_flag",
    "risk_category",
    "readmitted_30",
]


def validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    """Verify that all required database columns are available."""

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )


def load_database(
    replace_existing: bool = False,
) -> None:
    """Load curated encounter and prediction tables into PostgreSQL."""

    if not replace_existing:
        raise RuntimeError(
            "Database loading replaces the existing encounters and "
            "readmission_predictions tables. "
            "Call load_database(replace_existing=True) to proceed."
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES_PATH}. "
            "Run the modeling pipeline first."
        )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PREDICTIONS_PATH}. "
            "Run the modeling pipeline first."
        )

    print("Reading processed encounter data...")
    encounters = pd.read_csv(FEATURES_PATH)

    print("Reading prediction data...")
    predictions = pd.read_csv(PREDICTIONS_PATH)

    validate_columns(
        encounters,
        ENCOUNTER_COLUMNS,
        "Encounter dataset",
    )

    validate_columns(
        predictions,
        PREDICTION_COLUMNS,
        "Prediction dataset",
    )

    encounters = encounters[
        ENCOUNTER_COLUMNS
    ].copy()

    predictions = predictions[
        PREDICTION_COLUMNS
    ].copy()

    predictions = predictions.rename(
        columns={
            "readmitted_30": "actual_readmitted_30"
        }
    )

    print(f"Encounter rows: {len(encounters):,}")
    print(
        f"Encounter columns: {len(encounters.columns)}"
    )
    print(f"Prediction rows: {len(predictions):,}")
    print(
        f"Prediction columns: {len(predictions.columns)}"
    )

    engine = get_engine()

    try:
        print("Loading encounters table...")

        load_dataframe_to_table(
            encounters,
            "encounters",
            engine,
        )

        print(
            "Loading readmission_predictions table..."
        )

        load_dataframe_to_table(
            predictions,
            "readmission_predictions",
            engine,
        )

    finally:
        engine.dispose()

    print(
        "PostgreSQL database load completed successfully."
    )


if __name__ == "__main__":
    load_database()