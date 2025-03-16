# Reddit Topic Clustering

A Python tool that scrapes Reddit posts from specified subreddits, clusters them into coherent topics using NLP techniques, and generates human-readable topic names and descriptions using LLMs.

![Example dashboard](https://raw.githubusercontent.com/CharlesJub/Subreddit-Topic-Clustering/refs/heads/main/streamlit_example.png)

## Features

- **Reddit Data Retrieval**: Scrape posts and comments from any subreddit
- **Natural Language Processing**: Process and clean text data for analysis
- **Topic Modeling**: Identify meaningful topic clusters with BERTopic
- **LLM Summaries**: Generate concise topic names and descriptions using Ollama's LLM integration
- **Interactive Visualization**: Explore topics and posts through an intuitive Streamlit web interface

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/reddit-topic-clustering.git
   cd reddit-topic-clustering
   ```
2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your Reddit API credentials:

   ```
   API_KEY=your_reddit_api_secret
   CLIENT_ID=your_reddit_client_id
   ```
4. Install Ollama and download the llama3.2 model:

   ```bash
   # Follow instructions at https://ollama.com/download
   ollama pull llama3.2:latest
   ```

## Usage

### Command Line Interface

Run the main script to analyze a subreddit:

```bash
python src/main.py
```

By default, this will analyze the "politics" subreddit with posts from the past month. To modify these settings, edit the parameters in `main.py`.

### Streamlit Web Interface

For a more interactive experience, run the Streamlit app:

```bash
streamlit run src/app.py
```

The Streamlit interface provides:

- Customizable subreddit selection
- Multiple sorting options (hot, top posts by time period, new)
- Adjustable post limit
- Choice of LLM models for topic summarization
- Interactive visualizations including:
  - Topic distribution charts
  - Word clouds for each topic
  - Expandable post and comment previews
  - Searchable data table

## Project Structure

```
src/
├── app.py                         # Streamlit web application
├── main.py                        # Command-line application logic
├── data_retrieval/
│   └── subreddit_scraper.py       # Reddit API handling
├── data_prep/
│   └── transform.py               # Text preprocessing
├── modeling/
│   └── clustering.py              # Topic modeling logic
└── summarization/
    └── topic_summarizer.py        # LLM-based topic summarization
```

## How It Works

1. **Data Collection**: Retrieves posts and comments from a specified subreddit
2. **Data Preparation**: Cleans and preprocesses text (removing stopwords, lemmatization, etc.)
3. **Topic Modeling**: Uses BERTopic with UMAP and HDBSCAN to cluster similar posts
4. **Topic Interpretation**: Employs Llama 3.2 to generate human-readable topic names and descriptions
5. **Visualization**: Presents results through interactive charts and tables (Streamlit interface)

## Configuration

The main parameters can be adjusted in `main.py` or through the Streamlit interface:

- **Subreddit**: Change the subreddit name (default: "politics")
- **Time Period**: Choose from "hot", "month", "year", "week", or "new"
- **LLM Model**: Select different Ollama models (default: "llama3.2:latest")
- **Clustering Parameters**: Adjust settings in `modeling/clustering.py`

## Example Output

```
LLM-Generated Topic Summaries:
================================================================================
Topic 0: Election Fraud Claims
Description: Discussion about alleged election fraud in various states and legal challenges.
--------------------------------------------------------------------------------
Topic 1: Voting Rights Legislation
Description: Debates over voting access, voter ID laws, and election reform bills.
--------------------------------------------------------------------------------
...

Sample Posts by Topic:
Election Fraud Claims: Trump continues to push election fraud narrative despite court losses
Voting Rights Legislation: Senate to vote on new voting rights bill next week
```
