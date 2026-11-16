#!/usr/bin/env python
# coding: utf-8

"""
NYC Yellow Taxi - Prefect + MLflow homework

Local setup (PowerShell):
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install prefect mlflow pandas pyarrow scikit-learn

Start MLflow in Terminal 1:
    mlflow server --host 127.0.0.1 --port 5000

Optional: start the Prefect UI/API in another terminal:
    prefect server start

Run the pipeline:
    python task3.py

The pipeline trains on March 2023 and validates on April 2023.
"""

# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import pickle
from pathlib import Path
import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd
import prefect
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print("Imports successful")

# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "nyc-taxi-experiment"
REGISTERED_MODEL_NAME = "nyc-yellow-taxi-duration-model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

MODELS_FOLDER = Path("models")
MODELS_FOLDER.mkdir(parents=True, exist_ok=True)

print("MLflow tracking URI:", MLFLOW_TRACKING_URI)
print("MLflow experiment:", MLFLOW_EXPERIMENT_NAME)

# ============================================================
# QUESTION 1: SELECT THE TOOL
# ============================================================

# We are using Prefect as our workflow orchestrator.
# Prefect helps us create, manage, and execute ML pipelines.

print("Orchestrator used: Prefect")

# ============================================================
# QUESTION 2: VERSION
# ============================================================

# Print the installed version of Prefect.
print("Prefect Version:", prefect.__version__)

# ============================================================
# QUESTION 3: CREATING A PIPELINE
# ============================================================

@task(name="load-yellow-taxi-data")
def load_dataframe(year: int, month: int) -> pd.DataFrame:
    """
    Download and load the requested Yellow Taxi Parquet file.
    Prints the raw record count before data preparation.
    """
    url = (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/"
        f"yellow_tripdata_{year}-{month:02d}.parquet"
    )

    print(f"Loading Yellow Taxi data: {year}-{month:02d}")
    df = pd.read_parquet(url)

    print(f"Total records loaded ({year}-{month:02d}): {len(df):,}")
    return df


# ============================================================
# QUESTION 4: DATA PREPARATION
# ============================================================

@task(name="prepare-taxi-data")
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the homework's data-preparation logic:
      1. Calculate trip duration.
      2. Convert duration to minutes.
      3. Keep trips between 1 and 60 minutes.
      4. Convert pickup/drop-off location IDs to strings.
    """
    df = df.copy()

    df["duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    )
    df["duration"] = df["duration"].dt.total_seconds() / 60

    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)].copy()

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    print(f"Records after duration filtering: {len(df):,}")
    return df


# ============================================================
# QUESTION 5: FEATURE TRANSFORMATION AND MODEL TRAINING
# ============================================================

# @task(name="create-features")
# def create_X(df: pd.DataFrame, dv: DictVectorizer = None):
#     """
#     Keep pickup and drop-off locations as separate categorical features.
#     Do not create a combined PU_DO feature.
#     """
#     categorical = ["PULocationID", "DOLocationID"]
#     numerical = ["trip_distance"]

#     dicts = df[categorical + numerical].to_dict(orient="records")

#     if dv is None:
#         dv = DictVectorizer(sparse=True)
#         X = dv.fit_transform(dicts)
#     else:
#         X = dv.transform(dicts)

#     return X, dv

@task(name="create-features")
def create_X(df, dv=None):

    categorical = ['PULocationID', 'DOLocationID']

    dicts = df[categorical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


@task(name="train-linear-regression-and-log-to-mlflow")
def train_model(X_train, y_train, X_val, y_val, dv):
    """
    Fit default LinearRegression, print its intercept, and log the
    trained model and DictVectorizer artifact to MLflow.

    Returns both the DictVectorizer and trained LinearRegression model.
    """
    with mlflow.start_run() as active_run:
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Homework question 5: print model intercept.
        print("Model intercept:", model.intercept_)

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("training_dataset", "Yellow Taxi March 2023")
        mlflow.log_param("validation_dataset", "Yellow Taxi April 2023")
        mlflow.log_metric("validation_rmse", rmse)
        mlflow.log_metric("model_intercept", float(model.intercept_))

        # Save and log the fitted DictVectorizer separately.
        preprocessor_path = MODELS_FOLDER / "preprocessor.b"
        with preprocessor_path.open("wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(
            str(preprocessor_path),
            name="preprocessor"
        )

        # Log the trained sklearn model and register a model version.
        # The model is trained on the feature matrix produced by dv;
        # the DictVectorizer is logged above as a separate artifact.
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME
        )

        print("MLflow run ID:", active_run.info.run_id)
        print("Logged model URI:", model_info.model_uri)

        # Save run ID locally for reference.
        Path("run_id.txt").write_text(
            active_run.info.run_id,
            encoding="utf-8"
        )

        return dv, model


# ============================================================
# MAIN PREFECT PIPELINE
# ============================================================

@flow(name="nyc-yellow-taxi-training-pipeline")
def main_flow():
    """
    Orchestrates the end-to-end homework pipeline.

    Training data: March 2023 Yellow Taxi
    Validation data: April 2023 Yellow Taxi
    """

    # Load March 2023 training data.
    df_train_raw = load_dataframe(2023, 3)

    # Load April 2023 validation data.
    df_val_raw = load_dataframe(2023, 4)

    # Prepare and filter both datasets.
    df_train = prepare_dataframe(df_train_raw)
    df_val = prepare_dataframe(df_val_raw)

    # Fit DictVectorizer on training data and transform validation data.
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    # Target variable.
    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    # Train model and log artifacts/model to MLflow.
    dv, model = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        dv
    )

    print("Prefect pipeline completed successfully!")
    return model


# ============================================================
# PIPELINE EXECUTION
# ============================================================

if __name__ == "__main__":
    print("Starting NYC Yellow Taxi Prefect pipeline...")
    main_flow()
