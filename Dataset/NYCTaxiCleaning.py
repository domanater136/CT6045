import pandas as pd

RANDOM_SEED = 6942067
SAMPLE_PER_FILE = 12_500

taxi_files = [
    "NYCTaxi/yellow_tripdata_2015-01.csv",
    "NYCTaxi/yellow_tripdata_2016-01.csv",
    "NYCTaxi/yellow_tripdata_2016-02.csv",
    "NYCTaxi/yellow_tripdata_2016-03.csv"
]

samples = []

for file in taxi_files:
    print(f"Loading File: {file}")
    df = pd.read_csv(file)

    if 'RateCodeID' in df.columns: # Whoever did this >:(
        df.rename(columns={'RateCodeID': 'RatecodeID'}, inplace=True)

    # print all column names
    print("Columns in this file:", df.columns.tolist())

    # drop missing values
    df.dropna(inplace=True)

    # sample
    sample = df.sample(n=SAMPLE_PER_FILE, random_state=RANDOM_SEED)
    samples.append(sample)

# combine
nyc_taxi_sample = pd.concat(samples, ignore_index=True)
nyc_taxi_sample.to_csv("nyc_taxi.csv", index=False, encoding='utf-8')

