"""
Data preparation for the Wellness Tourism Package pipeline.

Loads the registered dataset from the repository data folder, removes columns
with no predictive value, fixes a label typo, removes duplicate and repeat
customer records, and writes a stratified train/test split to disk. The
workflow uploads the four CSV files as an artifact, so the training job
receives exactly these splits.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "tourism_project" / "data" / "tourism.csv"
TARGET = "ProdTaken"
TEST_SIZE = 0.20
RANDOM_STATE = 42
# Columns that describe the customer rather than one particular sales contact
PROFILE_COLUMNS = [
    "Age", "TypeofContact", "CityTier", "Occupation", "Gender", "MaritalStatus",
    "Designation", "MonthlyIncome", "Passport", "OwnCar",
]


def header(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    df = pd.read_csv(DATA_PATH)
    n_rows, n_cols = df.shape

    # 1. Remove columns with no predictive value. "Unnamed: 0" is a pandas
    #    index saved into the CSV and CustomerID is a unique identifier, so
    #    neither generalizes to new customers. ProductPitched is fully
    #    determined by Designation in this data, so it adds no information;
    #    Designation is kept because it is known about a customer before any
    #    pitch is made. The assertion stops the pipeline if new data ever
    #    breaks the one-to-one mapping that this decision relies on.
    pairs = df[["ProductPitched", "Designation"]].drop_duplicates()
    assert pairs["ProductPitched"].nunique() == len(pairs) == pairs["Designation"].nunique(), \
        "ProductPitched no longer maps one-to-one to Designation; revisit the column drop."
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")] + ["CustomerID", "ProductPitched"]
    df = df.drop(columns=drop_cols)

    # 2. "Fe Male" is a split spelling of "Female"
    n_typo = int((df["Gender"] == "Fe Male").sum())
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # 3. Remove exact duplicate rows
    n_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # 4. Collapse repeat records of the same customer. These rows agree on every
    #    profile column, including monthly income to the unit, and always share
    #    the same outcome. Keeping them would let copies of one customer fall on
    #    both sides of the split, which rewards memorization and overstates
    #    test performance. The first record of each customer is kept.
    groups = df.groupby(PROFILE_COLUMNS)[TARGET].agg(["size", "nunique"])
    repeat_groups = groups[groups["size"] > 1]
    n_conflicts = int((repeat_groups["nunique"] > 1).sum())
    n_repeats = int(df.duplicated(PROFILE_COLUMNS).sum())
    df = df.drop_duplicates(PROFILE_COLUMNS).reset_index(drop=True)

    header("Data cleaning")
    print(f"Loaded           : {DATA_PATH.relative_to(ROOT).as_posix()} ({n_rows} rows, {n_cols} columns)")
    print(f"Dropped columns  : {', '.join(drop_cols)}")
    print(f"Gender relabeled : {n_typo} 'Fe Male' values set to 'Female'")
    print(f"Exact duplicates : {n_dupes} removed")
    print(f"Repeat customers : {len(repeat_groups)} profiles recorded more than once, "
          f"{n_conflicts} with conflicting labels, {n_repeats} extra rows removed")
    print(f"Cleaned shape    : {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Gender levels    : {df['Gender'].value_counts().to_dict()}")

    # "Unmarried" is also missing from the data dictionary, but its purchase
    # rate differs from "Single", so the two levels are kept apart
    header("Purchase rate by MaritalStatus level")
    rates = df.groupby("MaritalStatus")[TARGET].agg(["mean", "size"]).sort_values("mean", ascending=False)
    rates.columns = ["purchase_rate", "rows"]
    print(rates.round(4).to_string())

    # 5. A stratified split keeps the purchase share equal in train and test
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    for name, frame in {"Xtrain": Xtrain, "Xtest": Xtest, "ytrain": ytrain, "ytest": ytest}.items():
        frame.to_csv(ROOT / f"{name}.csv", index=False)

    header("Train/test split (80/20, stratified on ProdTaken)")
    print(f"Xtrain : {Xtrain.shape}, purchase rate {ytrain.mean():.4f}")
    print(f"Xtest  : {Xtest.shape}, purchase rate {ytest.mean():.4f}")
    print("Files written: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")


if __name__ == "__main__":
    main()
