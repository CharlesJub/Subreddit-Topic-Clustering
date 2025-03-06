import os

import praw

client_id = os.environ.get("CLIENT_ID")
api_key = os.environ.get("API_KEY")

print(client_id)

# reddit = praw.Reddit(
#     client_id=client_id,
#     client_secret=api_key,
#     user_agent="SubTopicClustering:V1.0",
# )

# print(reddit.read_only)
