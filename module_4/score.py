import os
import uuid
import pickle
import sys
import pandas as pd

import mlflow

from sklearn. feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline



def generate_uuids(n):
    ride_ids = []
    for i in range(n):
      ride_ids.append(str(uuid.uuid4()))
    return ride_ids

def read_dataframe(filename: str):
    df = pd.read_parquet(filename)
    print(df.head())
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df= df[(df.duration >=1) & (df.duration <= 60)]
    df['ride_id'] = generate_uuids(len(df))
    
    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_'+ df['DOLocationID']
    categorical = ['PU_DO' ]
    numerical = ['trip_distance' ]
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts

def load_model(run_id):
    # logged_model = f's3://mlflow-models-alexey/1/{run_id}/artifacts/model'
    logged_model = f'runs:/{run_id}/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model

def apply_model(input_file, run_id, output_file):
    print(f"Applying model {run_id} to data from {input_file} and saving results to {output_file}")
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)
    print("loading the model...")
    model = load_model(run_id)
    y_pred = model.predict(dicts)
    print("Mean predictedduration:", y_pred.mean())

    # print("df.columns.tolist():", df.columns.tolist())

    df_result = pd.DataFrame()
    # print("df result",df_result)
    df_result['ride_id'] = df['ride_id']
    df_result['tpep_pickup_datetime'] = df['tpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID' ]
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']

    df_result['model_version'] = run_id

    # print("df_result.columns.tolist():", df_result.columns.tolist())
    df_result.to_parquet(output_file, engine='pyarrow', index=False)
    print(f"Saved results to {output_file}. Shape: {df_result.shape}")


def run():
    taxi_type = sys.argv[1] # 'yellow'
    year = int(sys.argv[2]) # 2023
    month = int(sys.argv[3]) # 3
    run_id = sys.argv[4] #'a2132cc2abbc4aa4ba19f14f6ccd1e9d'

    input_file  = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    print(f"Input file: {input_file}")
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'
    print(f"Output file: {output_file}")
    os.makedirs(f"output/{taxi_type}", exist_ok=True)

    apply_model(
    input_file=input_file,
    run_id=run_id,
    output_file=output_file
    )
if __name__ == '__main__':
    run()