import pandas as pd

RANDOM_SEED = 6942067
SAMPLE_SIZE = 50_000

# read CSVs safely
train = pd.read_csv("AmazonReviews/train.csv")
test = pd.read_csv("AmazonReviews/test.csv")

# raaaaagh
columns = ['label', 'review_title', 'review_text']
train.columns = columns
test.columns = columns

# combine train + test
full_reviews = pd.concat([train, test], ignore_index=True)

# add column names for Hive/Pyspark
full_reviews.rename(columns={
    'polarity': 'label',
    'title': 'review_title',
    'text': 'review_text'
}, inplace=True)

# drop any bad rows (missing polarity or text)
full_reviews.dropna(subset=['label', 'review_text'], inplace=True)

# extract sample
reviews_sample = full_reviews.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
reviews_sample.to_csv("amazon_reviews_reduced.csv", index=False, encoding='utf-8')
