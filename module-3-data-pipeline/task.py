#!/usr/bin/env python
# coding: utf-8

# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================
import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow
print("import successful")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)
print("mlflow setup successful")

# ============================================================
# QUESTION 1: SELECT THE TOOL
# ============================================================
# We are using Prefect as our workflow orchestrator.
# Prefect helps us create, manage, and execute ML pipelines.
# ============================================================

import prefect
print("Orchestrator used: Prefect")


# ============================================================
# QUESTION 2: VERSION
# ============================================================
# Print the installed version of Prefect.
# This confirms which version of the orchestrator we are using.
# ============================================================

print("Prefect Version:", prefect.__version__)


# ============================================================
# QUESTION 3: CREATING A PIPELINE
# ============================================================
# Task 1:
# Download the March 2023 Yellow Taxi dataset.
# Command used
#python task3.py --year 2023 --month 03
#
# Task 2:
# Read the parquet file using Pandas.
#
# Task 3:
# Print the total number of records loaded.
#
# Expected output should match one of the homework options.
# ============================================================


def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


# ============================================================
# QUESTION 4: DATA PREPARATION
# ============================================================
# Task 1:
# Calculate trip duration using pickup and dropoff timestamps.
#
# Task 2:
# Convert duration into minutes.
#
# Task 3:
# Filter trips where duration is between 1 and 60 minutes.
#
# Task 4:
# Convert pickup and dropoff location IDs into strings.
#
# Task 5:
# Print the final number of records after filtering.
# ============================================================


def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id



# ============================================================
# QUESTION 5: TRAIN A MODEL
# ============================================================
# Task 1:
# Select pickup and dropoff location IDs separately.
#
# Task 2:
# Convert the categorical features into dictionaries.
#
# Task 3:
# Fit a DictVectorizer.
#
# Task 4:
# Train LinearRegression with default parameters.
#
# Task 5:
# Print the model intercept.
#
# Task 6:
# Return both the DictVectorizer and trained model.
# ============================================================

def run(year, month):
    df_train = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id


# ============================================================
# QUESTION 6: REGISTER / LOG THE MODEL USING MLFLOW
# ============================================================
# Task 1:
# Create an MLflow experiment.
#
# Task 2:
# Start an MLflow run.
#
# Task 3:
# Combine DictVectorizer and LinearRegression into a pipeline.
#
# Task 4:
# Log the trained model using MLflow.
#
# Task 5:
# Print the model URI and model_size_bytes.
#
# Task 6:
# Open MLflow UI to inspect the logged model.
# ============================================================

if __name__ == "__main__":
    print("inside main")
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)

    with open("run_id.txt", "w") as f:
        f.write(run_id)













# ============================================================
# MAIN PREFECT PIPELINE
# ============================================================
# This is the main workflow.
#
# Execution order:
#
# 1. Load March 2023 Yellow Taxi data.
# 2. Prepare and filter the data.
# 3. Train the Linear Regression model.
# 4. Log the model using MLflow.
#
# Prefect manages the execution of these tasks.
# ============================================================


# ============================================================
# PIPELINE EXECUTION
# ============================================================
# Run the complete Prefect pipeline locally.
# ============================================================

