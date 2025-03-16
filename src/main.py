import modeling.clustering as model
from data_prep.transform import create_corpus, preprocess
from data_retrieval.subreddit_scraper import scrape_subreddit_posts
from summarization.topic_summarizer import (
    generate_topic_summaries,
    print_topic_summaries,
)

if __name__ == "__main__":
    # Scrape the data
    df = scrape_subreddit_posts("politics", "month")

    # Create and preprocess the corpus
    df["text"] = df.apply(create_corpus, axis=1)
    df["text"] = df["text"].apply(preprocess)

    # Create the topic model
    topic_model = model.create_models()

    # Fit and transform the data
    topics, probs = model.fit_transform_topics(topic_model, df["text"])

    # Assign topics to the DataFrame
    df = model.assign_topics_to_dataframe(df, topics)

    print("\nGenerating LLM Summaries for Topics:")
    topic_summaries = generate_topic_summaries(
        topic_model, df, llm_model="llama3.2:latest"
    )
    print_topic_summaries(topic_summaries)

    # Add the topic names to your dataframe
    topic_id_to_name = {
        topic_id: summary["name"] for topic_id, summary in topic_summaries.items()
    }
    df["topic_name"] = df["topic"].map(
        lambda x: topic_id_to_name.get(x, "Outlier") if x != -1 else "Outlier"
    )

    # Example: Print posts by topic name
    print("\nSample Posts by Topic:")
    for name in df["topic_name"].unique():
        sample = df[df["topic_name"] == name].iloc[0]
        print(f"{name}: {sample['title']}")
