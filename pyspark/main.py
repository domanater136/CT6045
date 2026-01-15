from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, unix_timestamp, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

# Start Spark
spark = SparkSession.builder.appName("TaxiFare").getOrCreate()

# Read the data
taxi_data = spark.read.csv("nyc_taxi.csv", header=True, inferSchema=True)

# Add trip duration
clean_data = taxi_data.withColumn(
    "trip_duration",
    (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60
)

# add pickup hour
clean_data = clean_data.withColumn("pickup_hour", hour("tpep_pickup_datetime"))
# add pickup day
clean_data = clean_data.withColumn("pickup_day", dayofweek("tpep_pickup_datetime"))

# features
core_features = ["pickup_hour", "pickup_day", "Trip_distance"]
# optional / noisy features
extra_features = ["Passenger_count", "trip_duration", "RateCodeID", "Payment_type"]

include_optional = False
# Construct feature set
if include_optional:
    what_to_use = core_features + extra_features
else:
    what_to_use = core_features

# double triple check
ready_data = clean_data.select(what_to_use + ["Fare_amount"]).dropna()

# put features together for the model
feature_maker = VectorAssembler(inputCols=what_to_use, outputCol="features")
data_with_features = feature_maker.transform(ready_data)
selected_data = data_with_features.select("features", "Fare_amount")
model_data = selected_data.withColumnRenamed("Fare_amount", "label")

# train and test
train, test = model_data.randomSplit([0.7, 0.3])

print("=================")
print("LINEAR REGRESSION")
print("=================")

# linear regression
lin_reg = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.1,
    elasticNetParam=0.0
)

lin_reg_model = lin_reg.fit(train)

# predictions
lin_reg_predictions = lin_reg_model.transform(test)

# how good it is
rmse_checker = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
mae_checker = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
r2_checker = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

lin_rmse = rmse_checker.evaluate(lin_reg_predictions) # RMSE Calculations
lin_mae = mae_checker.evaluate(lin_reg_predictions) # MAE Calculations
lin_r2 = r2_checker.evaluate(lin_reg_predictions) # R2 Calculations

print(f"RMSE error: ${lin_rmse:.2f}")
print(f"MAE error: ${lin_mae:.2f}")
print(f"R2 score: {lin_r2:.4f}")
print()

print("What each feature does in linear regression:")
for i, feature_name in enumerate(what_to_use):
    print(f"  {feature_name:20s}: {lin_reg_model.coefficients[i]:8.4f}")
print(f"  Starting point (intercept): {lin_reg_model.intercept:.4f}")
print()

print("=================")
print("GRADIENT BOOSTED TREES")
print("=================")

# gradient boosted trees
gbt_trees = GBTRegressor(
    featuresCol="features",
    labelCol="label",
    maxIter=50,
    maxDepth=5,
    stepSize=0.1,
    subsamplingRate=0.8
)

gbt_model = gbt_trees.fit(train)

# predictions with trees
gbt_predictions = gbt_model.transform(test)

# how good trees are
gbt_rmse = rmse_checker.evaluate(gbt_predictions) # RMSE Calculations
gbt_mae = mae_checker.evaluate(gbt_predictions) # MAE Calculations
gbt_r2 = r2_checker.evaluate(gbt_predictions) # R2 Calculations

print(f"RMSE error: ${gbt_rmse:.2f}")
print(f"MAE error: ${gbt_mae:.2f}")
print(f"R2 score: {gbt_r2:.4f}")
print()

# what features matter for trees
tree_importances = gbt_model.featureImportances.toArray()
importance_lookup = {what_to_use[i]: tree_importances[i] for i in range(len(what_to_use))}
importance_sorted = sorted(importance_lookup.items(), key=lambda x: x[1], reverse=True)

print("Most important features for trees:")
for feature, importance in importance_sorted:
    print(f"  {feature:20s}: {importance:.4f}")
print()

print("=================")
print("WHICH MODEL IS BETTER?")
print("=================")

# charts
fig, three_charts = plt.subplots(1, 3, figsize=(15, 4))

# chart 1: RMSE
three_charts[0].bar(['Linear', 'Trees'], [lin_rmse, gbt_rmse], color=['blue', 'orange'])
three_charts[0].set_title('Root mean square Error (lower is better)')
three_charts[0].set_ylabel('Dollars')
three_charts[0].text(0, lin_rmse, f'${lin_rmse:.2f}')
three_charts[0].text(1, gbt_rmse, f'${gbt_rmse:.2f}')

# chart 2: MAE
three_charts[1].bar(['Linear', 'Trees'], [lin_mae, gbt_mae], color=['blue', 'orange'])
three_charts[1].set_title('Mean absolute Error (lower is better)')
three_charts[1].set_ylabel('Dollars')
three_charts[1].text(0, lin_mae, f'${lin_mae:.2f}')
three_charts[1].text(1, gbt_mae, f'${gbt_mae:.2f}')

# chart 3: R2
three_charts[2].bar(['Linear', 'Trees'], [lin_r2, gbt_r2], color=['blue', 'orange'])
three_charts[2].set_title('R-Squared (higher is better)')
three_charts[2].set_ylabel('Score')
three_charts[2].set_ylim(0, 1)
three_charts[2].text(0, lin_r2, f'{lin_r2:.4f}')
three_charts[2].text(1, gbt_r2, f'{gbt_r2:.4f}')

plt.tight_layout()
plt.show()

# feature importance chart
sorted_by_importance = sorted(importance_lookup.items(), key=lambda x: x[1], reverse=True)
feature_names, importance_values = zip(*sorted_by_importance)

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importance_values, color='red')
plt.xlabel('How important')
plt.title('What matters for predicting taxi fare')
for i, val in enumerate(importance_values):
    plt.text(val, i, f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.show()
spark.stop()