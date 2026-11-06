# Wellness Tourism Package Purchase Prediction

MLOps pipeline that predicts whether a "Visit with Us" customer will buy the
Wellness Tourism Package. Every push to `main` runs three GitHub Actions jobs:
dataset registration, data preparation, and model training with MLflow
experiment tracking. The training job commits the tuned model back to `main`,
and Streamlit Community Cloud serves the app from
`tourism_project/deployment/app.py`, picking up each model the pipeline commits.

## Repository structure

```
.github/workflows/pipeline.yml           GitHub Actions workflow
tourism_project/
    data/tourism.csv                     registered dataset
    model_building/data_register.py      schema validation and data-quality summary
    model_building/prep.py               cleaning and stratified train/test split
    model_building/train.py              XGBoost tuning with MLflow tracking
    deployment/app.py                    Streamlit app
    deployment/requirements.txt          app dependencies
    deployment/best_tourism_model.joblib model committed by the pipeline
    deployment/model_meta.json           threshold and input schema committed by the pipeline
    requirements.txt                     pipeline dependencies
```

## Pipeline jobs

1. `register-dataset`: validates the expected columns, prints a data-quality
   summary, and uploads the registered CSV as an artifact.
2. `data-prep`: cleans the data and uploads the train and test splits as an
   artifact for the next job.
3. `model-training`: downloads the splits, tunes XGBoost with GridSearchCV while
   logging every trial to MLflow, evaluates the selected model, and commits it to
   `main`.
