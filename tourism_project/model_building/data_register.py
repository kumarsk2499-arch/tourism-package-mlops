"""
Dataset registration for the Wellness Tourism Package pipeline.

Reads tourism.csv from the repository data folder, confirms that every column in
the data dictionary is present, and prints a data-quality summary. A missing
column raises an error, which fails the GitHub Actions job and keeps every
downstream job from running on a malformed file.
"""
import hashlib
from pathlib import Path

import pandas as pd

# Paths resolve from the repository root, so the script behaves the same when
# it is launched from the notebook or from the workflow runner
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "tourism_project" / "data" / "tourism.csv"

TARGET = "ProdTaken"
ID_COLUMNS = ["CustomerID"]
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
# Category levels named in the data dictionary; anything else gets flagged
DICTIONARY_LEVELS = {
    "Gender": {"Male", "Female"},
    "MaritalStatus": {"Single", "Married", "Divorced"},
}
# Columns that describe the customer rather than one particular sales contact
PROFILE_COLUMNS = [
    "Age", "TypeofContact", "CityTier", "Occupation", "Gender", "MaritalStatus",
    "Designation", "MonthlyIncome", "Passport", "OwnCar",
]


def header(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    df = pd.read_csv(DATA_PATH)

    # Schema validation: stop the pipeline if any expected column is missing
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    feature_cols = [c for c in df.columns if c not in ID_COLUMNS + extra_cols]

    # A content hash records exactly which version of the file was registered
    sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()

    header("Dataset registration")
    print(f"Source file   : {DATA_PATH.relative_to(ROOT).as_posix()}")
    print(f"SHA-256       : {sha256}")
    print(f"Rows          : {df.shape[0]}")
    print(f"Columns       : {df.shape[1]}")
    print(f"Schema check  : passed, all {len(EXPECTED_COLUMNS)} expected columns present")
    print(f"Extra columns : {extra_cols if extra_cols else 'none'}")

    header("Data quality")
    quality = [
        ("Missing values", int(df.isna().sum().sum())),
        ("Duplicate CustomerID values", int(df["CustomerID"].duplicated().sum())),
        ("Duplicate rows (ID columns ignored)", int(df[feature_cols].duplicated().sum())),
    ]
    quality += [(f"{TARGET} = {label}", f"{n} rows ({n / len(df):.2%})")
                for label, n in df[TARGET].value_counts().sort_index().items()]
    for label, value in quality:
        print(f"{label:<36}: {value}")

    header("Repeat-customer check (exact duplicates set aside)")
    unique_rows = df[feature_cols].drop_duplicates()
    profiles = unique_rows.groupby(PROFILE_COLUMNS)[TARGET].agg(["size", "nunique"])
    repeats = profiles[profiles["size"] > 1]
    for label, value in [
        ("Customer profiles recorded more than once", len(repeats)),
        ("Rows involved", int(repeats["size"].sum())),
        (f"Profiles with conflicting {TARGET} values", int((repeats["nunique"] > 1).sum())),
    ]:
        print(f"{label:<43}: {value}")
    in_repeats = unique_rows[unique_rows.duplicated(PROFILE_COLUMNS, keep=False)]
    print("Number of repeat profiles in which each remaining column varies:")
    other_cols = [c for c in feature_cols if c not in PROFILE_COLUMNS + [TARGET]]
    varying = {c: int((in_repeats.groupby(PROFILE_COLUMNS)[c].nunique() > 1).sum()) for c in other_cols}
    for col, n in sorted(varying.items(), key=lambda kv: -kv[1]):
        print(f"  {col:<25}: {n}")

    header("Categorical levels")
    for col in df.select_dtypes(include="object").columns:
        levels = ", ".join(f"{k} ({v})" for k, v in df[col].value_counts().items())
        print(f"{col:<15}: {levels}")

    header("Levels not listed in the data dictionary")
    for col, allowed in DICTIONARY_LEVELS.items():
        unexpected = sorted(set(df[col].unique()) - allowed)
        print(f"{col:<15}: {unexpected if unexpected else 'none'}")

    header("Redundancy check")
    pairs = df[["ProductPitched", "Designation"]].drop_duplicates()
    one_to_one = pairs["ProductPitched"].nunique() == len(pairs) == pairs["Designation"].nunique()
    print(f"ProductPitched maps one-to-one to Designation: {one_to_one}")
    print(pairs.sort_values("ProductPitched").to_string(index=False))

    header("Numeric ranges")
    numeric = df.select_dtypes(include="number").drop(columns=ID_COLUMNS + extra_cols + [TARGET])
    print(numeric.describe().T[["min", "25%", "50%", "75%", "max"]].round(2).to_string())

    print("\nDataset registered successfully.")


if __name__ == "__main__":
    main()
