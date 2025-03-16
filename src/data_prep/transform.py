import re

import contractions
import emoji
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def create_corpus(row):
    """
    Create a corpus from a row of data.

    This function combines the title, post text, and comments from a given row
    into a single string, filtering out any None values.

    Args:
        row (dict): A dictionary containing 'title', 'post text', and 'comments' keys.

    Returns:
        str: A single string representing the corpus for the given row.
    """
    components = [row["title"], str(row["post text"]), *row["comments"]]
    return " ".join(filter(None, components))


def clean_markdown(text):
    # Remove bold/italic/strikethrough
    text = re.sub(r"\*{1,3}|_{1,3}|~{2}", "", text)
    # Convert markdown links to plain text
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    return text


def remove_emoji(text):
    return emoji.demojize(text, delimiters=(" ", " "))


def expand_contractions(text):
    return contractions.fix(text)


lemmatizer = WordNetLemmatizer()
custom_stopwords = set(stopwords.words("english")).union(
    {"reddit", "subreddit", "mod", "op", "http", "https"}
)


def preprocess(text):
    text = clean_markdown(text)
    text = remove_emoji(text)
    text = expand_contractions(text)
    text = re.sub(r"\W", " ", text.lower())
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in custom_stopwords and len(word) > 2
    ]
    return " ".join(tokens)
