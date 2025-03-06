import os

import praw
from dotenv import load_dotenv


def api_connect():
    """
    Establishes a connection to the Reddit API using credentials stored in environment variables.

    Returns:
        praw.Reddit: A Reddit API connection object.

    Required Environment Variables:
        API_KEY: The secret API key for Reddit API authentication.
        CLIENT_ID: The client ID for Reddit API authentication.
    """
    load_dotenv()
    # Grab secret values from .env file
    api_key = os.getenv("API_KEY")
    client_id = os.getenv("CLIENT_ID")

    reddit_conn = praw.Reddit(
        client_id=client_id,
        client_secret=api_key,
        user_agent="SubTopicClustering:V1.0",
    )

    return reddit_conn


if __name__ == "__main__":
    reddit_conn = api_connect()
    print(reddit_conn.read_only)
