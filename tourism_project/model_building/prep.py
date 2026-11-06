# Step 2 of the pipeline: data preparation.
# Loads the registered csv from the repo data folder, cleans it, and saves a
# stratified train/test split. In github actions the 4 csv files get uploaded
# as an artifact so the training job uses exactly these splits.
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "tourism_project" / "data" / "tourism.csv"
TARGET = "ProdTaken"
TEST_SIZE = 0.20
RANDOM_STATE = 42
# same profile columns as data_register.py, used to find repeat customers
PROFILE_COLUMNS = [
    "Age", "TypeofContact", "CityTier", "Occupation", "Gender", "MaritalStatus",
    "Designation", "MonthlyIncome", "Passport", "OwnCar",
]


def header(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    df = pd.read_csv(DATA_PATH)
    n_rows, n_cols = df.shape

    # 1. drop columns that can't help predict anything.
    # "Unnamed: 0" is a saved index and CustomerID is just an id.
    # ProductPitched is decided by Designation in this data, so it's redundant. Keeping
    # Designation since it's known before the pitch. The assert is there in case new data
    # breaks the one-to-one mapping, then this drop needs another look.
    pairs = df[["ProductPitched", "Designation"]].drop_duplicates()
    assert pairs["ProductPitched"].nunique() == len(pairs) == pairs["Designation"].nunique(), \
        "ProductPitched no longer maps one-to-one to Designation; revisit the column drop."
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")] + ["CustomerID", "ProductPitched"]
    df = df.drop(columns=drop_cols)

    # 2. "Fe Male" is a typo for Female
    n_typo = int((df["Gender"] == "Fe Male").sum())
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # 3. exact duplicate rows
    n_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # 4. repeat records of the same customer. They match on every profile column
    # (income to the unit) and always have the same outcome. If kept, copies of one
    # customer could land in both train and test and the test score would look better
    # than it really is. Keeping the first record per customer.
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

    # "Unmarried" isn't in the data dictionary either, checking if it behaves like
    # "Single" before deciding whether to merge them
    header("Purchase rate by MaritalStatus level")
    rates = df.groupby("MaritalStatus")[TARGET].agg(["mean", "size"]).sort_values("mean", ascending=False)
    rates.columns = ["purchase_rate", "rows"]
    print(rates.round(4).to_string())

    # 5. stratified split so train and test have the same share of buyers
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
