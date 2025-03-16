from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


def create_models():
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    hdbscan_model = HDBSCAN(
        min_cluster_size=10,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer = CountVectorizer(
        stop_words="english", ngram_range=(1, 1), max_features=2000
    )

    umap_model = UMAP(n_neighbors=5, min_dist=0.05, n_components=5, metric="cosine")

    topic_model = BERTopic(
        embedding_model=sentence_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        nr_topics=10,
        min_topic_size=8,
        top_n_words=6,
        verbose=False,
    )

    return topic_model


def fit_transform_topics(topic_model, clean_data):
    topics, probs = topic_model.fit_transform(clean_data)
    return topics, probs


def assign_topics_to_dataframe(df, topics):
    df["topic"] = topics
    return df


def print_topic_info(topic_model):
    topic_info = topic_model.get_topic_info()
    print(topic_info.head(10))


def print_topic_terms(topic_model, topics):
    # Convert topics to a set of unique topic IDs
    unique_topics = sorted(list(set(topics)))
    for topic_id in unique_topics:
        if topic_id != -1:  # Skip the outlier topic (-1)
            print(f"Topic {topic_id}:")
            print(topic_model.get_topic(topic_id))
            print()


def print_topic_counts(df):
    topic_counts = df["topic"].value_counts().sort_index()
    print(topic_counts)
