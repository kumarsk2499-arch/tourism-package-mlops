# Step 3 of the pipeline: model training with experiment tracking.
# Loads the splits from the data-prep job, tunes XGBoost with GridSearchCV, logs
# every parameter combo to MLflow, picks a threshold, evaluates on the test split,
# and saves the model next to the streamlit app so the workflow can commit it.
import json
import os
from pathlib import Path

# otherwise mlflow prints a link for every single nested run
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

# 3 x 5 x 2 x 2 x 2 = 120 combos
PARAM_GRID = {
    "xgbclassifier__n_estimators": [100, 200, 300],   # number of trees
    "xgbclassifier__max_depth": [4, 6, 8, 10, 12],    # deeper = more feature interactions
    "xgbclassifier__learning_rate": [0.05, 0.1],      # how much each tree adds
    "xgbclassifier__colsample_bytree": [0.6, 0.8],    # share of columns each tree sees
    "xgbclassifier__reg_lambda": [1, 5],              # L2 penalty on leaf weights
}


def header(title):
    print(f"\n{title}\n{'-' * len(title)}")


def short(params):
    # strip the "xgbclassifier__" prefix for logging
    return {k.replace("xgbclassifier__", ""): v for k, v in params.items()}


def one_standard_error_rule(cv_results):
    # used as refit= in GridSearchCV. Out of the candidates within one standard error
    # of the best mean CV F1, return the simplest one: lowest depth first, then fewest
    # trees, and if those tie, the higher score
    scores = np.asarray(cv_results["mean_test_score"])
    std_errors = np.asarray(cv_results["std_test_score"]) / np.sqrt(N_FOLDS)
    top = int(np.argmax(scores))
    eligible = np.flatnonzero(scores >= scores[top] - std_errors[top])
    return int(min(eligible, key=lambda i: (cv_results["param_xgbclassifier__max_depth"][i],
                                            cv_results["param_xgbclassifier__n_estimators"][i],
                                            -scores[i])))


def evaluate(y_true, proba, threshold):
    # metrics for the purchase class at the given threshold
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
    # tracking uri comes from the env: the workflow starts a server on port 5000,
    # and the notebook starts one locally
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    # in github actions these csv files come from the data-splits artifact
    Xtrain = pd.read_csv(ROOT / "Xtrain.csv")
    Xtest = pd.read_csv(ROOT / "Xtest.csv")
    ytrain = pd.read_csv(ROOT / "ytrain.csv").squeeze("columns")
    ytest = pd.read_csv(ROOT / "ytest.csv").squeeze("columns")
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert set(Xtrain.columns) == expected, f"Unexpected features: {sorted(set(Xtrain.columns) ^ expected)}"

    # class imbalance: weight buyers by non-buyers / buyers
    scale_pos_weight = float((ytrain == 0).sum() / (ytrain == 1).sum())

    # numeric cols pass through (trees don't care about scale), categoricals get
    # one-hot encoded inside the pipeline so the app can send plain text labels.
    # handle_unknown="ignore" so an unseen label doesn't crash the app
    preprocessor = make_column_transformer(
        ("passthrough", NUMERIC_FEATURES),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        verbose_feature_names_out=False,
    )
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,  # 1 thread per model, GridSearchCV runs the candidates in parallel
    )
    model_pipeline = make_pipeline(preprocessor, xgb_model)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    with mlflow.start_run(run_name="xgb-grid-search") as parent_run:
        # scoring on F1 for the purchase class: balances wasted calls (precision)
        # against missed buyers (recall), and isn't inflated by the 81% non-buyers
        # like accuracy is
        grid_search = GridSearchCV(model_pipeline, PARAM_GRID, scoring="f1", cv=cv, n_jobs=-1,
                                   refit=one_standard_error_rule, return_train_score=True)
        grid_search.fit(Xtrain, ytrain)
        results = grid_search.cv_results_
        chosen = grid_search.best_index_
        top = int(np.argmax(results["mean_test_score"]))

        # one nested run per combo so they can be compared side by side in mlflow
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

        # threshold from out-of-fold predictions on the training data,
        # the test split isn't touched until the final evaluation
        oof_proba = cross_val_predict(best_model, Xtrain, ytrain, cv=cv, method="predict_proba")[:, 1]
        thresholds = np.round(np.arange(0.20, 0.81, 0.01), 2)
        oof_f1 = [f1_score(ytrain, (oof_proba >= t).astype(int)) for t in thresholds]
        threshold = float(thresholds[int(np.argmax(oof_f1))])
        oof_f1_default = f1_score(ytrain, (oof_proba >= 0.5).astype(int))

        train_metrics, _ = evaluate(ytrain, best_model.predict_proba(Xtrain)[:, 1], threshold)
        test_proba = best_model.predict_proba(Xtest)[:, 1]
        test_metrics, test_pred = evaluate(ytest, test_proba, threshold)

        # parent run: selected params, search settings, threshold and all the metrics
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
        # tag the commit and run id when this runs inside github actions
        for key in ("GITHUB_SHA", "GITHUB_RUN_ID"):
            if os.environ.get(key):
                mlflow.set_tag(key.lower(), os.environ[key])

        # gain importance, one-hot columns added back up to their original column
        names = best_model.named_steps["columntransformer"].get_feature_names_out()
        gains = pd.Series(best_model.named_steps["xgbclassifier"].feature_importances_, index=names)
        source = {n: next((c for c in CATEGORICAL_FEATURES if n.startswith(f"{c}_")), n) for n in names}
        importance = gains.groupby(source).sum().sort_values(ascending=False)
        importance = importance / importance.sum()

        # save model + the metadata the app needs (threshold, column order, valid levels, ranges)
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

        # load the saved file back and make sure it gives the same predictions
        reload_ok = np.allclose(joblib.load(MODEL_PATH).predict_proba(Xtest)[:, 1], test_proba)

    # ---- summary for the notebook / workflow log ----
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
    # best validation score at each depth, with its train-fold score next to it
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
