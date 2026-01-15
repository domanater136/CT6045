REGISTER /usr/hdp/current/pig-client/lib/piggybank.jar;

raw_reviews =
    LOAD '/user/ct6045/data/amazon'
    USING PigStorage(',')
    AS (polarity:int, title:chararray, review_text:chararray);

clean_reviews =
    FILTER raw_reviews BY
        polarity IS NOT NULL AND
        review_text IS NOT NULL AND
        SIZE(review_text) > 0;

combined_reviews =
    FOREACH clean_reviews GENERATE
        polarity,
        CONCAT(title, ' ', review_text) AS full_text;

STORE combined_reviews
    INTO '/user/ct6045/output/amazon_clean'
    USING PigStorage('\t');
