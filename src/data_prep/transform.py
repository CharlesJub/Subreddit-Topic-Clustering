import re

import emoji
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def create_corpus(row):
    """ """
    components = [
        row["Title"],
        str(row["Post Text"]),
        *row["Comments"],  # Unpack the list of comments
    ]
    return " ".join(filter(None, components))


def clean_text(posts: pd.Dataframe):
    posts["Corpus"] = posts.apply(create_corpus, axis=1)


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
