import os

import pandas as pd
import praw
from dotenv import load_dotenv
from praw.models import MoreComments


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


def get_comments(post, limit=10):
    return [comment.body for comment in post.comments[:limit] if not comment.stickied]


def scrape_subreddit_posts(
    subreddit: str, sort: str = "month", limit: int = 100
) -> pd.DataFrame:
    """
    Optimized Reddit post scraper with enhanced performance through:
    - Batch comment processing
    - Reduced API calls
    - Efficient memory management
    - Connection reuse
    """
    reddit_conn = api_connect()
    subreddit_obj = reddit_conn.subreddit(subreddit)

    # Configure sort parameters with safety buffer for stickied posts
    sort_config = {
        "hot": {"method": "hot", "limit": limit * 2},
        "month": {"method": "top", "time_filter": "month", "limit": limit * 2},
        "year": {"method": "top", "time_filter": "year", "limit": limit * 2},
        "week": {"method": "top", "time_filter": "week", "limit": limit * 2},
        "new": {"method": "new", "limit": limit * 2},
    }

    if sort not in sort_config:
        raise ValueError(
            f"Invalid sort option. Choose from: {', '.join(sort_config.keys())}"
        )

    # Batch fetch posts with optimized limit
    posts = getattr(subreddit_obj, sort_config[sort]["method"])(
        limit=sort_config[sort].get("limit"),
        **{k: v for k, v in sort_config[sort].items() if k not in ["method", "limit"]},
    )

    posts_data = []
    for idx, post in enumerate(posts, 1):
        if post.stickied:
            continue

        # Configure comment parameters before retrieval
        post.comment_limit = 15  # Get slightly more than needed for filtering
        post.comments.replace_more(limit=0)  # Prevent deep comment traversal

        # Batch process comments with generator
        comments = [
            comment.body
            for comment in post.comments
            if not comment.stickied and not isinstance(comment, MoreComments)
        ][
            :10
        ]  # Take up to 10 comments

        posts_data.append(
            {
                "Title": post.title,
                "Post Text": post.selftext,
                "ID": post.id,
                "Comments": comments,
            }
        )

        if len(posts_data) >= limit:
            break

    # Convert to DataFrame with specific dtype for Comments column
    df = pd.DataFrame(posts_data).astype({"Comments": "object"})

    # Standardize column names to lowercase
    df.columns = df.columns.str.lower()

    return df


if __name__ == "__main__":
    posts = scrape_subreddit_posts("politics", "hot")
    print(posts)
