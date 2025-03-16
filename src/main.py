import ollama
import pandas as pd

import modeling.clustering as model
from data_prep.transform import create_corpus, preprocess
from data_retrieval.subreddit_scraper import scrape_subreddit_posts


def generate_topic_summaries(topic_model, df, llm_model="llama3.2:latest"):
    """
    Use an LLM to generate descriptive names and summaries for each topic cluster.

    Parameters:
    - topic_model: The fitted BERTopic model
    - df: DataFrame containing the texts and their assigned topics
    - llm_model: The Ollama model to use

    Returns:
    - Dictionary mapping topic_ids to {'name': '...', 'description': '...'}
    """
    topic_summaries = {}

    # Get unique topics (excluding -1 which is the outlier topic)
    unique_topics = [topic for topic in df["topic"].unique() if topic != -1]

    for topic_id in unique_topics:
        # Get the top terms for this topic
        topic_terms = topic_model.get_topic(topic_id)
        terms_str = ", ".join([term for term, _ in topic_terms])

        # Get representative documents for this topic
        topic_docs = df[df["topic"] == topic_id]["text"].tolist()
        # Limit to a few examples to avoid context length issues
        example_docs = topic_docs[:5]
        docs_str = "\n- " + "\n- ".join(example_docs)

        # Create prompt for the LLM
        prompt = f"""
        I have a cluster of documents from a subreddit on a related topic. 
        
        The key terms for this topic are: {terms_str}
        
        Here are some example documents in this cluster:
        {docs_str}
        
        Based on these terms and examples, please provide:
        1. A concise, descriptive name for this topic (max 5 words)
        2. A brief one-sentence description of what this topic represents
        
        Format your response as:
        Name: [topic name]
        Description: [brief description]
        """

        # Send to Ollama LLM
        response = ollama.chat(
            model=llm_model, messages=[{"role": "user", "content": prompt}]
        )

        # Parse the response
        response_text = response["message"]["content"]

        # Extract name and description
        try:
            name_line = [
                line for line in response_text.split("\n") if line.startswith("Name:")
            ][0]
            desc_line = [
                line
                for line in response_text.split("\n")
                if line.startswith("Description:")
            ][0]

            name = name_line.replace("Name:", "").strip()
            description = desc_line.replace("Description:", "").strip()

            topic_summaries[topic_id] = {"name": name, "description": description}
        except IndexError:
            # Fallback if parsing fails
            topic_summaries[topic_id] = {
                "name": f"Topic {topic_id}",
                "description": response_text.strip(),
            }

    return topic_summaries


# Function to print the LLM-generated summaries
def print_topic_summaries(topic_summaries):
    """Print the LLM-generated topic summaries in a readable format"""
    print("\nLLM-Generated Topic Summaries:")
    print("=" * 80)
    for topic_id, summary in topic_summaries.items():
        print(f"Topic {topic_id}: {summary['name']}")
        print(f"Description: {summary['description']}")
        print("-" * 80)


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
