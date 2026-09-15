"""Load and clean the hospital readmission dataset."""

from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/diabetic_data.csv")
PROCESSED_DATA_PATH = Path("data/processed/diabetic_clean_stage1.csv")

ADMISSION_TYPE_MAP = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not Available",
    6: "NULL",
    7: "Trauma Center",
    8: "Not Mapped",
}

ADMISSION_SOURCE_MAP = {
    1: "Physician Referral",
    2: "Clinic Referral",
    3: "HMO Referral",
    4: "Transfer from Hospital",
    5: "Transfer from Skilled Nursing Facility",
    6: "Transfer from Another Health Care Facility",
    7: "Emergency Room",
    8: "Court/Law Enforcement",
    9: "Not Available",
    10: "Transfer from Critical Access Hospital",
    11: "Normal Delivery",
    13: "Other",
    14: "Extramural Birth",
    17: "NULL",
    20: "Not Mapped",
    22: "Transfer from Another Facility",
    25: "Transfer from Outpatient Surgery",
}

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw hospital encounter dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize missing-value representations."""

    cleaned = df.copy()

    # The original dataset represents missing values with '?'.
    cleaned = cleaned.replace("?", pd.NA)

    categorical_unknown = [
        "medical_specialty",
        "payer_code",
        "race",
        "diag_1",
        "diag_2",
        "diag_3",
    ]

    for column in categorical_unknown:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("Unknown")

    if "gender" in cleaned.columns:
        cleaned["gender"] = cleaned["gender"].replace(
            "Unknown/Invalid",
            "Unknown",
        )

    for column in ["max_glu_serum", "A1Cresult"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(
                "Not Measured"
            )

    return cleaned


def remove_high_missingness_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns excluded because of substantial missingness."""

    cleaned = df.copy()

    if "weight" in cleaned.columns:
        cleaned = cleaned.drop(columns=["weight"])

    return cleaned


def create_readmission_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary 30-day readmission target."""

    if "readmitted" not in df.columns:
        raise ValueError(
            "Expected 'readmitted' column was not found."
        )

    cleaned = df.copy()

    cleaned["readmitted_30"] = (
        cleaned["readmitted"] == "<30"
    ).astype(int)

    return cleaned

def add_admission_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Map admission type and admission source IDs to readable categories."""

    cleaned = df.copy()

    cleaned["admission_type"] = (
        cleaned["admission_type_id"].map(
            ADMISSION_TYPE_MAP
        )
    )

    cleaned["admission_source"] = (
        cleaned["admission_source_id"].map(
            ADMISSION_SOURCE_MAP
        )
    )

    return cleaned

def prepare_data(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Run the complete initial data-preparation workflow."""

    df = load_raw_data(input_path)
    df = clean_missing_values(df)
    df = remove_high_missingness_columns(df)
    df = add_admission_categories(df)
    df = create_readmission_target(df)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    return df


if __name__ == "__main__":
    prepared_df = prepare_data()

    print(
        f"Prepared dataset saved to: {PROCESSED_DATA_PATH}"
    )
    print(f"Rows: {len(prepared_df):,}")
    print(
        f"Columns: {len(prepared_df.columns):,}"
    )