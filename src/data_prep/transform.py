from data_retrieval.subreddit_scraper import scrape_subreddit_posts

if __name__ == "__main__":
    posts = scrape_subreddit_posts("politics", "month")
