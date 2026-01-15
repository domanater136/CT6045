USE ct6045;

CREATE EXTERNAL TABLE nyc_taxi (
    vendor_id INT,
    pickup_datetime STRING,
    dropoff_datetime STRING,
    passenger_count INT,
    trip_distance DOUBLE,
    pickup_longitude DOUBLE,
    pickup_latitude DOUBLE,
    rate_code INT,
    store_and_fwd STRING,
    dropoff_longitude DOUBLE,
    dropoff_latitude DOUBLE,
    payment_type INT,
    fare_amount DOUBLE,
    extra DOUBLE,
    mta_tax DOUBLE,
    improvement_surcharge DOUBLE,
    tip_amount DOUBLE,
    tolls_amount DOUBLE,
    total_amount DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/user/ct6045/data/taxi';

-- Fare-to-distance relationship
SELECT
    ROUND(trip_distance, 1) AS distance_bucket,
    AVG(fare_amount) AS avg_fare,
    COUNT(*) AS trip_count
FROM nyc_taxi
WHERE trip_distance > 0 AND fare_amount > 0
GROUP BY ROUND(trip_distance, 1)
ORDER BY distance_bucket;
