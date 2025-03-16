import ollama


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


def print_topic_summaries(topic_summaries):
    """Print the LLM-generated topic summaries in a readable format"""
    print("\nLLM-Generated Topic Summaries:")
    print("=" * 80)
    for topic_id, summary in topic_summaries.items():
        print(f"Topic {topic_id}: {summary['name']}")
        print(f"Description: {summary['description']}")
        print("-" * 80)
