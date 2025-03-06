import praw

reddit = praw.Reddit(
    client_id=secret.client_id,
    client_secret=secret.api_key,
    user_agent="SubTopicClustering:V1.0",
)

print(reddit.read_only)
