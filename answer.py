from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, split, explode

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Movie Recommendation System") \
    .getOrCreate()

# Read movies.csv and ratings.csv
movies_df = spark.read.option("header", "true").csv("movies.csv")
ratings_df = spark.read.option("header", "true").csv("ratings.csv")

# Step A: Apply ALS Algorithm
# Splitting dataset into train and test sets
(train, test) = ratings_df.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
als = ALS(maxIter=5, regParam=0.01, userCol="Userid", itemCol="movieid", ratingCol="Rating")
model = als.fit(train)

# Evaluate the model by computing RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="Rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root Mean Square Error (RMSE):", rmse)

# Step B: Count Unique Genres
unique_genres_df = movies_df.withColumn("genre", explode(split(col("genre"), "\\|"))) \
    .groupBy("genre").count()

unique_genres_df.show()

# Step C: Total Rating by Movie and Genre
total_rating_by_movie = ratings_df.groupBy("movieid").sum("Rating")
total_rating_by_genre = ratings_df.join(movies_df, "movieid") \
    .withColumn("genre", explode(split(col("genre"), "\\|"))) \
    .groupBy("genre").sum("Rating")

total_rating_by_movie.show()
total_rating_by_genre.show()

# Step D: Top and Worst Performing Movies
top_performing_movies = predictions.orderBy("prediction", ascending=False).limit(10)
worst_performing_movies = predictions.orderBy("prediction").limit(5)

top_performing_movies.show()
worst_performing_movies.show()

# Step E: Average Rating
average_rating_all_movies = ratings_df.groupBy().avg("Rating").collect()[0][0]
average_rating_per_user = ratings_df.groupBy("Userid").avg("Rating").agg({"avg(Rating)": "avg"}).collect()[0][0]

print("Average Rating of All Movies:", average_rating_all_movies)
print("Average Rating per User:", average_rating_per_user)

# Stop SparkSession
spark.stop()
