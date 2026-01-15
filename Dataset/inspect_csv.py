import pandas as pd
import os

def inspect_csv(path):
    print(f"File: {path}")
    print(f"File size: {os.path.getsize(path) / (1024**2):.2f} MB")

    df = pd.read_csv(path)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Column names:")
    print(df.columns.tolist())

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nFirst 3 rows:")
    print(df.head(3))
    print("=================================================")


# -------- Amazon Reviews --------
inspect_csv("AmazonReviews/test.csv")
inspect_csv("AmazonReviews/train.csv")

# -------- NYC Taxi (repeat for each file) --------
inspect_csv("NYCTaxi/yellow_tripdata_2015-01.csv")
inspect_csv("NYCTaxi/yellow_tripdata_2016-01.csv")
inspect_csv("NYCTaxi/yellow_tripdata_2016-02.csv")
inspect_csv("NYCTaxi/yellow_tripdata_2016-03.csv")
