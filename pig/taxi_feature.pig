raw_taxi =
    LOAD '/user/ct6045/data/taxi'
    USING PigStorage(',')
    AS (
        vendor_id:int,
        pickup_time:chararray,
        dropoff_time:chararray,
        passenger_count:int,
        trip_distance:double,
        pickup_longitude:double,
        pickup_latitude:double,
        rate_code:int,
        store_and_fwd:chararray,
        dropoff_longitude:double,
        dropoff_latitude:double,
        payment_type:int,
        fare_amount:double,
        extra:double,
        mta_tax:double,
        improvement_surcharge:double,
        tip_amount:double,
        tolls_amount:double,
        total_amount:double
    );

valid_trips =
    FILTER raw_taxi BY
        trip_distance > 0 AND
        fare_amount > 0 AND
        passenger_count > 0;

features =
    FOREACH valid_trips GENERATE
        trip_distance,
        fare_amount,
        (fare_amount / trip_distance) AS fare_per_mile,
        tip_amount,
        (tip_amount / fare_amount) AS tip_ratio;

STORE features
    INTO '/user/ct6045/output/taxi_features'
    USING PigStorage('\t');
