# Databricks notebook source
# MAGIC %md
# MAGIC ** GUARDIAN NEWSPAPER API**

# COMMAND ----------

# MAGIC %pip install textblob nltk

# COMMAND ----------

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# COMMAND ----------

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

# Create Spark session
spark = SparkSession.builder.appName("GuardianDataAnalysis").getOrCreate()

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# API request
api_key = "4a2d596f-53f5-47e7-8621-95f3665307d2"
url = f"https://content.guardianapis.com/search?api-key={api_key}&show-fields=wordcount,byline,bodyText"
response = requests.get(url)
data = response.json()["response"]["results"]

# Function to calculate sentiment
def get_sentiment_scores(text):
    blob = TextBlob(text)
    sia_scores = sia.polarity_scores(text)
    return {
        'sentiment': 'positive' if blob.sentiment.polarity > 0 else 
                     'negative' if blob.sentiment.polarity < 0 else 'neutral',
        'compound_score': sia_scores['compound']
    }

# Create DataFrame with sentiment analysis
df = pd.DataFrame([{
    "source": "The Guardian",
    "title": item["webTitle"],
    "date_posted": item["webPublicationDate"],
    "author": item["fields"].get("byline", "Unknown"),
    "url": item["webUrl"],
    "content": item["fields"].get("bodyText", ""),
    "word_count": item["fields"].get("wordcount", 0),
    **get_sentiment_scores(item["fields"].get("bodyText", ""))
} for item in data])

# Convert to Spark DataFrame
spark_df = spark.createDataFrame(df)

# Visualization: Sentiment Distribution
plt.figure(figsize=(10, 6))
sentiment_counts = df['sentiment'].value_counts()
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
plt.title("Sentiment Distribution")
plt.show()

# Visualization: Compound Score Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='compound_score', kde=True)
plt.title("Sentiment Compound Score Distribution")
plt.show()

# Display the DataFrame
display(spark_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS the_news;
# MAGIC CREATE TABLE IF NOT EXISTS the_news.news_table (
# MAGIC source STRING,
# MAGIC title STRING,
# MAGIC date_posted DATE,
# MAGIC author STRING,
# MAGIC url STRING,
# MAGIC content STRING,
# MAGIC word_count INT,
# MAGIC sentiment STRING,
# MAGIC compound_score DOUBLE
# MAGIC )

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType, IntegerType, DoubleType
import pandas as pd
from datetime import datetime

# Initialize Spark session
spark = SparkSession.builder.appName("CreateTableExample").getOrCreate()

# Define the schema explicitly
schema = StructType([
    StructField("source", StringType(), True),
    StructField("title", StringType(), True),
    StructField("date_posted", DateType(), True), 
    StructField("author", StringType(), True),
    StructField("url", StringType(), True),
    StructField("content", StringType(), True),
    StructField("word_count", IntegerType(), True),
    StructField("sentiment", StringType(), True),
    StructField("compound_score", DoubleType(), True)
])

# Create a sample pandas DataFrame with 10 rows
data = [
    ("The Guardian", "Climate Change", datetime.strptime("2023-01-01", "%Y-%m-%d").date(), "John Doe", "http://example1.com", "Climate content", 100, "neutral", 0.0),
    ("BBC", "Technology", datetime.strptime("2023-01-02", "%Y-%m-%d").date(), "Jane Smith", "http://example2.com", "Tech content", 150, "positive", 0.5),
    ("CNN", "Politics", datetime.strptime("2023-01-03", "%Y-%m-%d").date(), "Bob Johnson", "http://example3.com", "Politics content", 200, "negative", -0.3),
    ("Reuters", "Economy", datetime.strptime("2023-01-04", "%Y-%m-%d").date(), "Alice Brown", "http://example4.com", "Economy content", 180, "positive", 0.2),
    ("Al Jazeera", "World News", datetime.strptime("2023-01-05", "%Y-%m-%d").date(), "Mohammed Ali", "http://example5.com", "World news content", 220, "neutral", 0.1),
    ("New York Times", "Health", datetime.strptime("2023-01-06", "%Y-%m-%d").date(), "Sarah Lee", "http://example6.com", "Health content", 190, "positive", 0.4),
    ("The Economist", "Finance", datetime.strptime("2023-01-07", "%Y-%m-%d").date(), "David Chen", "http://example7.com", "Finance content", 250, "neutral", -0.1),
    ("National Geographic", "Science", datetime.strptime("2023-01-08", "%Y-%m-%d").date(), "Emily Wilson", "http://example8.com", "Science content", 300, "positive", 0.6),
    ("Time", "Culture", datetime.strptime("2023-01-09", "%Y-%m-%d").date(), "Michael Brown", "http://example9.com", "Culture content", 170, "neutral", 0.0),
    ("Wall Street Journal", "Business", datetime.strptime("2023-01-10", "%Y-%m-%d").date(), "Lisa Taylor", "http://example10.com", "Business content", 230, "positive", 0.3)
]

dataframe = pd.DataFrame(data, columns=["source", "title", "date_posted", "author", "url", "content", "word_count", "sentiment", "compound_score"])

# Convert pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(dataframe, schema=schema)

# Write the Spark DataFrame as a table
spark_df.write.mode('overwrite').saveAsTable('the_news.news_table')

# Display the table contents
display(spark.table('the_news.news_table'))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM the_news.news_table;
# MAGIC