import pandas as pd
import re
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import contractions

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
STOP_WORDS = set(stopwords.words('english'))


def load_data(path):
    dataset = pd.read_csv(path)
    dataset['review'] = dataset['review_title'].fillna('') + " " + dataset['review_text'].fillna('')
    dataset['polarity'] = dataset['label']
    return dataset


def clean_and_tokenize(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for token in tokens:
        if token not in STOP_WORDS:
            lemmatized = lemmatizer.lemmatize(token, pos='v')
            lemmatized_tokens.append(lemmatized)
    tokens = lemmatized_tokens
    cleaned_text = ' '.join(tokens)
    return cleaned_text, tokens


def apply_cleaning(dataset):
    results = dataset['review'].apply(clean_and_tokenize)

    dataset['clean_review'] = [result[0] for result in results]
    dataset['tokens'] = [result[1] for result in results]

    return dataset


def compute_word_scores(df):
    word_scores = defaultdict(int)

    for _, row in df.iterrows():
        # Use pre-tokenized tokens instead of tokenizing again
        tokens = row['tokens']

        for token in tokens:
            if row['polarity'] == 2:
                word_scores[token] += 1
            elif row['polarity'] == 1:
                word_scores[token] -= 1

    return word_scores


def split_sentiment(word_scores):
    positive_words = {}
    negative_words = {}

    for word, score in word_scores.items():
        if score > 0:
            positive_words[word] = score
        elif score < 0:
            negative_words[word] = abs(score)

    return positive_words, negative_words


def plot_wordcloud(word_dict, title):
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(word_dict)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()


def run_sentiment_pipeline(csv_path, clean_csv_path="amazon_clean_nlp.csv", use_prev_clean=False):
    if use_prev_clean:
        dataset = pd.read_csv(clean_csv_path)
    else:
        dataset = load_data(csv_path)
        dataset = apply_cleaning(dataset)
        dataset[['polarity', 'clean_review', 'tokens']].to_csv(clean_csv_path, index=False)

    # wordcloud
    scores = compute_word_scores(dataset)
    positive, negative = split_sentiment(scores)

    plot_wordcloud(positive, "Positive Words")
    plot_wordcloud(negative, "Negative Words")
    print("Most common positive words:", dict(list(positive.items())[:10]))
    print("Most common negative words:", dict(list(negative.items())[:10]))

    # review length analysis
    dataset['review_length'] = dataset['tokens'].apply(len)

    plt.figure(figsize=(8, 5))
    sns.boxplot(x='polarity', y='review_length', data=dataset)
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.title("Review Length by Sentiment")
    plt.ylabel("Number of Words")
    plt.show()


if __name__ == "__main__":
    run_sentiment_pipeline(csv_path="amazon_reviews.csv")