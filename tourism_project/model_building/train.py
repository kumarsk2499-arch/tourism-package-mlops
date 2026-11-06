"""
Model training with experiment tracking for the Wellness Tourism Package pipeline.

Loads the train/test splits produced by the data-prep job, tunes an XGBoost
classifier with GridSearchCV, logs every parameter combination to MLflow as a
nested run, selects the final configuration with the one-standard-error rule,
picks a decision threshold from out-of-fold predictions on the training data,
evaluates the tuned model on the held-out test split, and saves the model next
to the Streamlit app so the workflow can commit it to the repository.
"""
import json
import os
from pathlib import Path

# Keep the log readable: MLflow otherwise prints a link for every nested run
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = ROOT / "tourism_project" / "deployment"
MODEL_PATH = DEPLOY_DIR / "best_tourism_model.joblib"
META_PATH = DEPLOY_DIR / "model_meta.json"
EXPERIMENT_NAME = "tourism-package-prediction"
RANDOM_STATE = 42
N_FOLDS = 5

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = ["TypeofContact", "Occupation", "Gender", "MaritalStatus", "Designation"]

PARAM_GRID = {
    "xgbclassifier__n_estimators": [100, 200, 300],   # number of boosting rounds
    "xgbclassifier__max_depth": [4, 6, 8, 10, 12],    # tree depth sets how many features can interact
    "xgbclassifier__learning_rate": [0.05, 0.1],      # shrinkage applied to each new tree
    "xgbclassifier__colsample_bytree": [0.6, 0.8],    # share of features sampled per tree
    "xgbclassifier__reg_lambda": [1, 5],              # L2 penalty on leaf weights
}


def header(title):
    print(f"\n{title}\n{'-' * len(title)}")


def short(params):
    """Drop the pipeline step prefix from parameter names."""
    return {k.replace("xgbclassifier__", ""): v for k, v in params.items()}


def one_standard_error_rule(cv_results):
    """Index of the least complex candidate whose mean CV F1 is within one
    standard error of the top mean. Complexity is ranked by tree depth, then by
    number of trees; ties go to the higher score."""
    scores = np.asarray(cv_results["mean_test_score"])
    std_errors = np.asarray(cv_results["std_test_score"]) / np.sqrt(N_FOLDS)
    top = int(np.argmax(scores))
    eligible = np.flatnonzero(scores >= scores[top] - std_errors[top])
    return int(min(eligible, key=lambda i: (cv_results["param_xgbclassifier__max_depth"][i],
                                            cv_results["param_xgbclassifier__n_estimators"][i],
                                            -scores[i])))


def evaluate(y_true, proba, threshold):
    """Classification metrics for the purchase class at a given threshold."""
    pred = (proba >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
    }
    return metrics, pred


def main():
    # The tracking URI comes from the environment: the workflow starts an MLflow
    # server on port 5000, and the notebook does the same during development
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    # The splits arrive from the data-prep job as a workflow artifact
    Xtrain = pd.read_csv(ROOT / "Xtrain.csv")
    Xtest = pd.read_csv(ROOT / "Xtest.csv")
    ytrain = pd.read_csv(ROOT / "ytrain.csv").squeeze("columns")
    ytest = pd.read_csv(ROOT / "ytest.csv").squeeze("columns")
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert set(Xtrain.columns) == expected, f"Unexpected features: {sorted(set(Xtrain.columns) ^ expected)}"

    # Class imbalance: weight the purchase class by the negative-to-positive ratio
    scale_pos_weight = float((ytrain == 0).sum() / (ytrain == 1).sum())

    # Numeric columns pass through unchanged because tree splits do not depend
    # on feature scale. Categorical columns are one-hot encoded inside the
    # pipeline, so the app can send raw category labels exactly as a user picks
    # them, and unseen labels are ignored instead of raising an error.
    preprocessor = make_column_transformer(
        ("passthrough", NUMERIC_FEATURES),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        verbose_feature_names_out=False,
    )
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,  # GridSearchCV runs candidates in parallel instead
    )
    model_pipeline = make_pipeline(preprocessor, xgb_model)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    with mlflow.start_run(run_name="xgb-grid-search") as parent_run:
        # F1 on the purchase class balances wasted calls (precision) against
        # missed buyers (recall), and unlike accuracy it is not inflated by the
        # majority class. Several deep configurations score within noise of
        # each other, so the refit step keeps the simplest one inside one
        # standard error of the top score.
        grid_search = GridSearchCV(model_pipeline, PARAM_GRID, scoring="f1", cv=cv, n_jobs=-1,
                                   refit=one_standard_error_rule, return_train_score=True)
        grid_search.fit(Xtrain, ytrain)
        results = grid_search.cv_results_
        chosen = grid_search.best_index_
        top = int(np.argmax(results["mean_test_score"]))

        # Every parameter combination becomes a nested run for side-by-side comparison
        for i, params in enumerate(results["params"]):
            with mlflow.start_run(run_name=f"trial-{i + 1:03d}", nested=True):
                mlflow.log_params(short(params))
                mlflow.log_metrics({
                    "cv_f1_mean": results["mean_test_score"][i],
                    "cv_f1_std": results["std_test_score"][i],
                    "cv_train_f1_mean": results["mean_train_score"][i],
                    "cv_rank": int(results["rank_test_score"][i]),
                })
                mlflow.set_tag("selected", str(i == chosen))

        best_model = grid_search.best_estimator_
        best_params = short(grid_search.best_params_)
        chosen_cv_f1 = float(results["mean_test_score"][chosen])

        # Decision threshold from out-of-fold training predictions, so the test
        # split stays untouched until the final evaluation
        oof_proba = cross_val_predict(best_model, Xtrain, ytrain, cv=cv, method="predict_proba")[:, 1]
        thresholds = np.round(np.arange(0.20, 0.81, 0.01), 2)
        oof_f1 = [f1_score(ytrain, (oof_proba >= t).astype(int)) for t in thresholds]
        threshold = float(thresholds[int(np.argmax(oof_f1))])
        oof_f1_default = f1_score(ytrain, (oof_proba >= 0.5).astype(int))

        train_metrics, _ = evaluate(ytrain, best_model.predict_proba(Xtrain)[:, 1], threshold)
        test_proba = best_model.predict_proba(Xtest)[:, 1]
        test_metrics, test_pred = evaluate(ytest, test_proba, threshold)

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_params({
            "scale_pos_weight": round(scale_pos_weight, 4),
            "cv_folds": N_FOLDS,
            "candidates": len(results["params"]),
            "selection_rule": "one standard error",
            "decision_threshold": threshold,
        })
        mlflow.log_metrics({
            "cv_f1_top": float(results["mean_test_score"][top]),
            "cv_f1_selected": chosen_cv_f1,
            "oof_f1_at_0_50": oof_f1_default,
            "oof_f1_at_threshold": max(oof_f1),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        })
        for key in ("GITHUB_SHA", "GITHUB_RUN_ID"):
            if os.environ.get(key):
                mlflow.set_tag(key.lower(), os.environ[key])

        # Gain-based importances, with one-hot levels summed back to their column
        names = best_model.named_steps["columntransformer"].get_feature_names_out()
        gains = pd.Series(best_model.named_steps["xgbclassifier"].feature_importances_, index=names)
        source = {n: next((c for c in CATEGORICAL_FEATURES if n.startswith(f"{c}_")), n) for n in names}
        importance = gains.groupby(source).sum().sort_values(ascending=False)
        importance = importance / importance.sum()

        # Save the model and the metadata the app needs to build valid inputs
        DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        meta = {
            "decision_threshold": threshold,
            "feature_order": list(Xtrain.columns),
            "categories": {c: sorted(Xtrain[c].unique().tolist()) for c in CATEGORICAL_FEATURES},
            "category_defaults": {c: Xtrain[c].mode()[0] for c in CATEGORICAL_FEATURES},
            "numeric_ranges": {c: {"min": float(Xtrain[c].min()), "max": float(Xtrain[c].max()),
                                   "median": float(Xtrain[c].median())} for c in NUMERIC_FEATURES},
            "best_params": best_params,
            "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
            "feature_importance": importance.round(4).to_dict(),
            "library_versions": {"scikit-learn": sklearn.__version__, "xgboost": xgb.__version__,
                                 "pandas": pd.__version__},
            "mlflow_run_id": parent_run.info.run_id,
        }
        META_PATH.write_text(json.dumps(meta, indent=2))
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.log_artifact(str(META_PATH), artifact_path="model")

        # Reload from disk to confirm the saved file reproduces the predictions
        reload_ok = np.allclose(joblib.load(MODEL_PATH).predict_proba(Xtest)[:, 1], test_proba)

    # -----------------------------------------------------------------------
    # Summary for the notebook or workflow log
    # -----------------------------------------------------------------------
    header("Training data")
    print(f"Train rows: {len(Xtrain)} | Test rows: {len(Xtest)} | Features: {Xtrain.shape[1]}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    header("Hyperparameter search (5-fold CV, scoring = F1)")
    print(f"Candidates evaluated   : {len(results['params'])} ({len(results['params']) * N_FOLDS} fits)")
    print(f"Top mean CV F1         : {results['mean_test_score'][top]:.4f} "
          f"(std error {results['std_test_score'][top] / np.sqrt(N_FOLDS):.4f}, "
          f"depth {results['param_xgbclassifier__max_depth'][top]}, "
          f"{results['param_xgbclassifier__n_estimators'][top]} trees)")
    print(f"Selected mean CV F1    : {chosen_cv_f1:.4f} (simplest candidate within one standard error)")
    print(f"Selected train-fold F1 : {results['mean_train_score'][chosen]:.4f}")
    print("Selected parameters:")
    for k, v in best_params.items():
        print(f"  {k:<17}: {v}")
    print("\nBest candidate at each tree depth (mean F1 across folds):")
    by_depth = (pd.DataFrame({"max_depth": results["param_xgbclassifier__max_depth"].astype(int),
                              "validation_f1": results["mean_test_score"],
                              "train_f1": results["mean_train_score"]})
                .sort_values("validation_f1", ascending=False)
                .groupby("max_depth").first().round(4))
    print(by_depth.to_string())

    header("Decision threshold (out-of-fold F1 on training data)")
    print(f"OOF F1 at 0.50   : {oof_f1_default:.4f}")
    print(f"Chosen threshold : {threshold:.2f} (OOF F1 {max(oof_f1):.4f})")

    header("Performance at the chosen threshold")
    print(pd.DataFrame({"train": train_metrics, "test": test_metrics}).T.round(4).to_string())

    header("Test set classification report")
    print(classification_report(ytest, test_pred, target_names=["Not purchased", "Purchased"], digits=4))

    header("Test set confusion matrix")
    cm = pd.DataFrame(confusion_matrix(ytest, test_pred),
                      index=["Actual: Not purchased", "Actual: Purchased"],
                      columns=["Predicted: Not purchased", "Predicted: Purchased"])
    print(cm.to_string())

    header("Feature importance (gain share, one-hot levels summed per column)")
    print(importance.round(4).to_string())

    header("Saved artifacts")
    print(f"Model         : {MODEL_PATH.relative_to(ROOT).as_posix()}")
    print(f"Metadata      : {META_PATH.relative_to(ROOT).as_posix()}")
    print(f"Reload check  : {'passed' if reload_ok else 'FAILED'} (saved model reproduces test predictions)")
    print(f"MLflow run id : {parent_run.info.run_id}")
    if not reload_ok:
        raise SystemExit("Saved model failed the reload check.")


if __name__ == "__main__":
    main()
