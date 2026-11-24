import pickle
import pandas as pd
import numpy as np
import os

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    print(f"Reading data from {filename}...")
    df = pd.read_parquet(filename)
    print(f"Data shape before filtering: {df.shape}")
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    print(f"Data shape after filtering: {df.shape}")
    return df

df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)
# 1 standard deviation of the predictions
standard_deviation = np.std(y_pred)
print('predicted standard deviation:', standard_deviation)

# 2 Preparing the output
year = 2023
month = 3
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
df_result = pd.DataFrame({
    'ride_id': df['ride_id'],
    'predictions': y_pred
})

output_file = 'predictions.parquet'

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

print(df_result.shape)
print(df_result.dtypes)
print(df_result.head())

size_bytes = os.path.getsize(output_file)
print("Size in MB:", size_bytes / 1_000_000)
print("Size in MiB:", size_bytes / (1024 * 1024))