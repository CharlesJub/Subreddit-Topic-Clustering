import os

import pandas as pd
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


def scrape_subreddit_posts(
    subreddit: str, sort: str = "month", limit: int = 50
) -> pd.DataFrame:
    """
    Scrape posts from a specified subreddit, excluding pinned posts.

    Args:
        subreddit (str): Name of the subreddit to scrape.
        sort (str): Sorting method for posts. Options: "month", "year", "week", "hot", "new". Default is "month".
        limit (int): Maximum number of posts to retrieve. Default is 50.

    Returns:
        pd.DataFrame: DataFrame containing post data.

    Raises:
        ValueError: If an invalid sort option is provided.
    """
    reddit_conn = api_connect()

    subreddit = reddit_conn.subreddit(subreddit)

    sort_options = {
        "hot": lambda: subreddit.hot(limit=None),
        "month": lambda: subreddit.top(time_filter="month", limit=None),
        "year": lambda: subreddit.top(time_filter="year", limit=None),
        "week": lambda: subreddit.top(time_filter="week", limit=None),
        "new": lambda: subreddit.new(limit=None),
    }

    if sort not in sort_options:
        raise ValueError(
            f"Invalid sort option. Choose from: {', '.join(sort_options.keys())}"
        )

    posts = sort_options[sort]()

    posts_data = []
    for post in posts:
        if not post.stickied:
            posts_data.append(
                {
                    "Title": post.title,
                    "Post Text": post.selftext,
                    "ID": post.id,
                    "Comments": [
                        comment.body
                        for comment in post.comments[:10]
                        if not comment.stickied
                    ],
                }
            )
        if len(posts_data) >= limit:
            break

    return pd.DataFrame(posts_data)


if __name__ == "__main__":
    posts = scrape_subreddit_posts("politics", "hot")
    print(posts)
