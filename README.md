Requirments:
Hortonworks Data Platform (HDP) version 2.6.5
In CT6045\data\DatasetReduction\ | Use ReviewsCleaning.py and NYCTaxiCleaning.py on AmazonReviews and NYCTaxi to create cleaned dataset

Required Directory Structure:
ct6045/upload
ct6045/data
ct6045/data/amazon_reviews.csv
ct6045/data/nyc_taxi.csv

ct6045/hive
ct6045/hive/amazon_polarity.hql
ct6045/hive/taxi_fare.hql

ct6045/pig
ct6045/pig/amazon_cleaning.pig
ct6045/pig/taxi_feature.pig

ct6045/mapreduce
ct6045/mapreduce/mapper.py
ct6045/mapreduce/reducer.py

HDFS Structure:
/user/ct6045/data/amazon
/user/ct6045/data/taxi
/user/ct6045/output

STEPS FOR REPRODUCABILITY:
1 - HDFS the directories:
hdfs dfs -mkdir -p /user/ct6045/data/amazon
hdfs dfs -mkdir -p /user/ct6045/data/taxi
hdfs dfs -mkdir -p /user/ct6045/output

2 - Upload raw data to HDFS
hdfs dfs -put data/amazon_reviews.csv /user/ct6045/data/amazon/
hdfs dfs -put data/nyc_taxi.csv /user/ct6045/data/taxi/

3 - Run PIG ETL and Hive Scripts
pig pig/amazon_cleaning.pig
pig pig/taxi_feature.pig
hive -f hive/amazon_polarity.hql
hive -f hive/taxi_fare.hql

4 - run MAP Reducer
hadoop jar /usr/hdp/current/hadoop-mapreduce-client/hadoop-streaming.jar -files mapper.py,reducer.py -mapper "python2 mapper.py" -reducer "python2 reducer.py" -input /user/ct6045/amazon/cleaned -output /user/ct6045/results/word_sentiment

5 - Run the Local Python Scripts using their Venv:
CT6045\nlp+sentament\nlp+sentament\main.py
CT6045\nlp+sentament\nlp+sentament\sentament.py
CT6045\Pyspark\main.py
CT6045\graphs\main/py



