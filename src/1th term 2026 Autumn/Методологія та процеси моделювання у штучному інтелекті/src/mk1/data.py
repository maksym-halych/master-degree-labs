"""Download, load and clean the Diabetes 130-US Hospitals dataset (UCI id 296).

One row is one hospital encounter of a diabetic patient, 1999-2008. The target
is readmission within 30 days of discharge, which about one encounter in eleven
ends with — the class imbalance this lab works against.
"""

import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mk1.config import DATASETS, SEED, TEST_SIZE

log = logging.getLogger(__name__)

URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
ARCHIVE = DATASETS / "diabetes-130.zip"
CSV_NAME = "diabetic_data.csv"

TARGET = "readmitted_30"

NUMERIC = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

AGE_ORDER = [f"[{lo}-{lo + 10})" for lo in range(0, 100, 10)]

# Every medication column of the source table, in its order.
ALL_MEDS = [
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

# Medications prescribed in at least 1% of encounters. The other fifteen are
# near-constant — two are never prescribed at all — and only feed the
# medication-count features built in mk1.features.
MEDS = [
    "metformin",
    "repaglinide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "insulin",
]

LOW_CARD = [
    "race",
    "gender",
    "admission_type_id",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
    *MEDS,
]

HIGH_CARD = [
    "diag_1",
    "diag_2",
    "diag_3",
    "medical_specialty",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
]

CATEGORICAL = [*LOW_CARD, *HIGH_CARD]

# ID codes that the mapping table (IDS_mapping.csv) describes as "NULL",
# "Not Available", "Not Mapped" or "Unknown/Invalid": missing values disguised
# as categories.
DISGUISED_MISSING = {
    "admission_type_id": {5, 6, 8},
    "discharge_disposition_id": {18, 25, 26},
    "admission_source_id": {9, 15, 17, 20, 21},
}

# Discharges to a hospice or ending in death: such a patient cannot be
# readmitted, so the label of these encounters is fixed by the discharge
# itself rather than by anything the model should learn.
NO_READMISSION_POSSIBLE = {11, 13, 14, 19, 20, 21}


def download() -> Path:
    """
    Fetch the UCI archive once and extract the encounters table.

    Returns:
        Path to the extracted CSV under the datasets cache.

    Raises:
        urllib.error.URLError: If the archive cannot be downloaded.
        KeyError: If the archive no longer holds the expected CSV.
    """
    csv_path = DATASETS / CSV_NAME
    if csv_path.exists():
        log.debug("Dataset already cached: %s", csv_path)
        return csv_path

    DATASETS.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", URL)
    # Written to a temporary name first, so an interrupted download never
    # leaves behind an archive that looks complete.
    partial = ARCHIVE.with_suffix(".part")
    with (
        urllib.request.urlopen(URL, timeout=120) as response,
        partial.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)
    partial.replace(ARCHIVE)

    with zipfile.ZipFile(ARCHIVE) as archive:
        csv_path.write_bytes(archive.read(CSV_NAME))
    log.info("Extracted %s (%.1f MB)", csv_path, csv_path.stat().st_size / 1e6)
    return csv_path


def load_raw() -> pd.DataFrame:
    """
    Read the encounters table exactly as published.

    The dataset marks a missing value with "?". The pandas default NA list is
    switched off because it also contains "None", which this dataset uses as a
    real category: "test not performed" in max_glu_serum and A1Cresult.

    Returns:
        One row per hospital encounter, all 50 columns.
    """
    return pd.read_csv(
        download(),
        na_values=["?"],
        keep_default_na=False,
        low_memory=False,
    )


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Turn the raw encounters into independent, labelled observations.

    Args:
        raw: The table as returned by load_raw().

    Returns:
        The cleaned frame (features plus the binary TARGET column) and the
        number of rows left after each step, for the report.
    """
    steps = {"raw": len(raw)}
    df = raw.copy()

    for column, codes in DISGUISED_MISSING.items():
        df[column] = df[column].where(~df[column].isin(codes))

    df = df[~df["discharge_disposition_id"].isin(NO_READMISSION_POSSIBLE)]
    steps["without_death_or_hospice"] = len(df)

    # A patient's encounters are not independent: the same person, with the
    # same history, would otherwise land on both sides of the train/test split.
    # Keeping only the first encounter is the convention of the dataset's
    # original study (Strack et al., 2014).
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    steps["first_encounter_per_patient"] = len(df)

    df = df[df["gender"] != "Unknown/Invalid"]
    steps["valid_gender"] = len(df)

    # The ID columns are codes, not quantities: as strings they cannot be
    # mistaken for numbers by any later step.
    for column in DISGUISED_MISSING:
        df[column] = df[column].map(lambda v: np.nan if pd.isna(v) else str(int(v)))

    df[TARGET] = (df["readmitted"] == "<30").astype(int)
    # weight is 97% missing: imputing it would mean inventing it.
    df = df.drop(columns=["encounter_id", "patient_nbr", "weight", "readmitted"])
    return df.reset_index(drop=True), steps


def load_clean() -> pd.DataFrame:
    """Raw table passed through clean(), without the step counts."""
    return clean(load_raw())[0]


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified hold-out split; the test part is touched only by the final run.

    Args:
        df: Cleaned frame holding TARGET.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    return X_train, X_test, y_train, y_test
