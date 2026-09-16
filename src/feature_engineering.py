"""Create reusable features for hospital readmission modeling."""

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/diabetic_clean_stage1.csv")
OUTPUT_PATH = Path("data/processed/diabetic_features.csv")


AGE_MAP = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

DISCHARGE_DISPOSITION_MAP = {
    1: "Discharged to Home",
    2: "Transferred to Short-Term Hospital",
    3: "Transferred to Skilled Nursing Facility",
    4: "Transferred to Intermediate Care Facility",
    5: "Transferred to Another Inpatient Facility",
    6: "Home with Home Health Care",
    7: "Left Against Medical Advice",
    8: "Home with IV Provider",
    9: "Admitted as Inpatient",
    10: "Neonate",
    11: "Expired",
    12: "Still Patient",
    13: "Hospice - Home",
    14: "Hospice - Medical Facility",
    15: "Transferred to Swing Bed",
    16: "Transferred to Inpatient Rehab",
    17: "Transferred to Long-Term Care Hospital",
    18: "Expired",
    19: "Expired",
    20: "Expired",
    21: "Expired",
    22: "Transferred to Inpatient Rehab",
    23: "Transferred to Long-Term Care Hospital",
    24: "Transferred to Nursing Facility",
    25: "Not Mapped",
    26: "Unknown",
    27: "Transferred to Inpatient Rehab",
    28: "Transferred to Federal Health Care Facility",
}

MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


def categorize_diagnosis(code: object) -> str:
    """Map ICD-style diagnosis codes to broad clinical categories."""

    if pd.isna(code) or code == "Unknown":
        return "Unknown"

    code = str(code).strip()

    if code.startswith("V"):
        return "Supplementary"

    if code.startswith("E"):
        return "Injury/External Cause"

    try:
        code_num = float(code)
    except ValueError:
        return "Other"

    if 390 <= code_num <= 459:
        return "Circulatory"
    if 460 <= code_num <= 519:
        return "Respiratory"
    if 520 <= code_num <= 579:
        return "Digestive"
    if 580 <= code_num <= 629:
        return "Genitourinary"
    if 630 <= code_num <= 679:
        return "Pregnancy"
    if 680 <= code_num <= 709:
        return "Skin"
    if 710 <= code_num <= 739:
        return "Musculoskeletal"
    if 740 <= code_num <= 759:
        return "Congenital"
    if 760 <= code_num <= 779:
        return "Perinatal"
    if 780 <= code_num <= 799:
        return "Symptoms/Signs"
    if 800 <= code_num <= 999:
        return "Injury/Poisoning"
    if 140 <= code_num <= 239:
        return "Neoplasms"
    if 240 <= code_num <= 279:
        return "Endocrine/Diabetes"
    if 280 <= code_num <= 289:
        return "Blood Disorders"
    if 290 <= code_num <= 319:
        return "Mental Disorders"
    if 320 <= code_num <= 359:
        return "Neurological"

    return "Other"

def add_discharge_disposition(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Map discharge disposition IDs to readable categories."""

    result = df.copy()

    result["discharge_disposition"] = (
        result["discharge_disposition_id"]
        .map(DISCHARGE_DISPOSITION_MAP)
        .fillna("Unknown")
    )

    return result

def add_age_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Convert age ranges into numeric midpoint values."""

    result = df.copy()
    result["age_numeric"] = result["age"].map(AGE_MAP)

    return result


def add_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create prior healthcare utilization features."""

    result = df.copy()

    utilization_columns = [
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
    ]

    for column in utilization_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).fillna(0)

    result["total_prior_visits"] = (
        result["number_outpatient"]
        + result["number_emergency"]
        + result["number_inpatient"]
    )

    result["prior_inpatient_visit"] = (
        result["number_inpatient"] > 0
    ).astype(int)

    return result


def add_medication_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create medication-related features used during analysis."""

    result = df.copy()

    available_columns = [
        column
        for column in MEDICATION_COLUMNS
        if column in result.columns
    ]

    if available_columns:
        result["medication_count"] = (
            result[available_columns]
            .fillna("No")
            .ne("No")
            .sum(axis=1)
        )
    else:
        result["medication_count"] = 0

    if "change" in result.columns:
        result["medication_changed"] = (
            result["change"] == "Ch"
        ).astype(int)
    else:
        result["medication_changed"] = 0

    if "diabetesMed" in result.columns:
        result["diabetes_medication"] = (
            result["diabetesMed"] == "Yes"
        ).astype(int)
    else:
        result["diabetes_medication"] = 0

    return result


def add_diagnosis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create broad diagnosis categories and diagnosis indicators."""

    result = df.copy()

    diagnosis_columns = [
        "diag_1",
        "diag_2",
        "diag_3",
    ]

    for column in diagnosis_columns:
        category_column = f"{column}_category"

        result[category_column] = result[column].apply(
            categorize_diagnosis
        )

    category_columns = [
        "diag_1_category",
        "diag_2_category",
        "diag_3_category",
    ]

    result["diabetes_diagnosis"] = (
        result[category_columns]
        .eq("Endocrine/Diabetes")
        .any(axis=1)
        .astype(int)
    )

    result["diagnosis_category_count"] = (
        result[category_columns]
        .nunique(axis=1)
    )

    return result


def add_complexity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create encounter-complexity indicators used in exploratory analysis."""

    result = df.copy()

    if "num_medications" in result.columns:
        medication_75th = result["num_medications"].quantile(
            0.75
        )

        result["high_medication_burden"] = (
            result["num_medications"] >= medication_75th
        ).astype(int)

    if "time_in_hospital" in result.columns:
        los_75th = result["time_in_hospital"].quantile(
            0.75
        )

        result["long_stay"] = (
            result["time_in_hospital"] >= los_75th
        ).astype(int)
    return result


def engineer_features(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Run the complete feature-engineering workflow."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    df = add_discharge_disposition(df)
    df = add_age_feature(df)
    df = add_utilization_features(df)
    df = add_medication_features(df)
    df = add_diagnosis_features(df)
    df = add_complexity_features(df)

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
    engineered_df = engineer_features()

    print(
        f"Feature-engineered dataset saved to: {OUTPUT_PATH}"
    )
    print(f"Rows: {len(engineered_df):,}")
    print(
        f"Columns: {len(engineered_df.columns):,}"
    )