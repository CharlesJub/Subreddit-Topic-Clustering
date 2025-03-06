import re

import emoji
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from data_retrieval.subreddit_scraper import scrape_subreddit_posts

# posts = scrape_subreddit_posts("hockey", "week")
# posts.to_csv("temp.csv")
post = pd.read_csv("temp.csv")
post["Comments"] = post["Comments"].apply(
    eval
)  # Convert string representation of list back to list


def create_corpus(row):
    """
    Create a corpus from a row of data.

    This function combines the title, post text, and comments from a given row
    into a single string, filtering out any None values.

    Args:
        row (dict): A dictionary containing 'Title', 'Post Text', and 'Comments' keys.

    Returns:
        str: A single string representing the corpus for the given row.
    """
    components = [row["Title"], str(row["Post Text"]), *row["Comments"]]
    return " ".join(filter(None, components))


post["Corpus"] = post.apply(create_corpus, axis=1)


def normalize_text(row):
    # Remove emojis
    clean_text = emoji.replace_emoji(row["Corpus"], replace="")

    # Convert to lowercase
    clean_text = clean_text.lower()

    # Remove punctuation
    clean_text = re.sub(r"[^\w\s]", "", clean_text)

    lemmatizer = WordNetLemmatizer()

    # Tokenize the text
    words = word_tokenize(clean_text)

    # Remove stop words and lemmatize
    base_stopwords = set(stopwords.words("english"))
    clean_words = [
        lemmatizer.lemmatize(word, pos="v")
        for word in words
        if word not in base_stopwords
    ]

    # Join the words back into a string
    clean_text = " ".join(clean_words)

    return clean_text


post["Clean"] = post.apply(normalize_text, axis=1)

from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_vectorize(corpus_col):

    vectorizer = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2), stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(corpus_col)

    return tfidf_matrix


print(tfidf_vectorize(post["Corpus"]))
