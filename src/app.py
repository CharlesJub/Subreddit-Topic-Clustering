import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

import modeling.clustering as model
from data_prep.transform import create_corpus, preprocess
from data_retrieval.subreddit_scraper import scrape_subreddit_posts
from summarization.topic_summarizer import generate_topic_summaries

# Set page config
st.set_page_config(
    page_title="Reddit Topic Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4500;  /* Reddit orange */
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #676363;
        border-radius: 4px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4500 !important;
        color: white !important;
    }
    .topic-card {
        background-color: #676363;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #eee;
        margin-bottom: 1rem;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# App title and description
st.markdown(
    '<h1 class="main-header">Reddit Topic Explorer</h1>', unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">Discover the main topics being discussed in any subreddit using advanced NLP techniques.</p>',
    unsafe_allow_html=True,
)

# Sidebar for inputs
with st.sidebar:
    st.markdown('<p class="sidebar-header">Settings</p>', unsafe_allow_html=True)

    subreddit = st.text_input("Subreddit", "television")

    sort_options = {
        "Top posts this month": "month",
        "Top posts this week": "week",
        "Top posts this year": "year",
        "Hot posts": "hot",
        "New posts": "new",
    }
    sort_option = st.selectbox("Sort by", list(sort_options.keys()))

    post_limit = st.slider("Number of posts to analyze", 20, 300, 150, step=10)

    llm_options = ["llama3.2:latest", "deepseek-r1:1.5b"]
    llm_model = st.selectbox("LLM for topic summarization", llm_options)

    analyze_button = st.button("Analyze Subreddit", use_container_width=True)

# Main app functionality
if analyze_button:
    try:
        # Show progress
        progress_bar = st.progress(0)

        with st.spinner(f"Scraping data from r/{subreddit}..."):
            df = scrape_subreddit_posts(
                subreddit, sort_options[sort_option], post_limit
            )
            st.success(f"Successfully scraped {len(df)} posts from r/{subreddit}")
            progress_bar.progress(25)

        # Process data
        with st.spinner("Processing text data..."):
            df["text"] = df.apply(create_corpus, axis=1)
            df["text"] = df["text"].apply(preprocess)
            progress_bar.progress(50)

        # Create model and analyze topics
        with st.spinner("Analyzing topics..."):
            topic_model = model.create_models()
            topics, probs = model.fit_transform_topics(topic_model, df["text"])
            df = model.assign_topics_to_dataframe(df, topics)
            progress_bar.progress(75)

        # Generate topic summaries
        with st.spinner("Generating topic summaries with LLM..."):
            topic_summaries = generate_topic_summaries(
                topic_model, df, llm_model=llm_model
            )

            # Add topic names to dataframe
            topic_id_to_name = {
                topic_id: summary["name"]
                for topic_id, summary in topic_summaries.items()
            }
            df["topic_name"] = df["topic"].map(
                lambda x: topic_id_to_name.get(x, "Outlier") if x != -1 else "Outlier"
            )
            progress_bar.progress(100)

        # Display results
        st.markdown("## 📊 Analysis Results")

        # Topic overview
        st.markdown("### 🔍 Topic Overview")
        topic_counts = df["topic_name"].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            # Display topic counts as a bar chart using Plotly
            fig = px.bar(
                x=topic_counts.index,
                y=topic_counts.values,
                labels={"x": "Topic", "y": "Number of Posts"},
                title="Number of Posts per Topic",
                color=topic_counts.values,
                color_continuous_scale="Viridis",
            )
            fig.update_layout(
                xaxis_title="Topic",
                yaxis_title="Count",
                coloraxis_showscale=False,
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Display topic distribution as a pie chart using Plotly
            fig = px.pie(
                values=topic_counts.values,
                names=topic_counts.index,
                title="Topic Distribution",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Topic details
        st.markdown("### 📑 Topic Details")
        tabs = st.tabs([f"{name} ({count})" for name, count in topic_counts.items()])

        for i, (topic_name, count) in enumerate(topic_counts.items()):
            with tabs[i]:
                col1, col2 = st.columns([1, 2])

                with col1:
                    # Get the topic ID for this topic name
                    topic_id = next(
                        (
                            tid
                            for tid, name in topic_id_to_name.items()
                            if name == topic_name
                        ),
                        -1,
                    )

                    if topic_id != -1:
                        st.markdown(f"#### {topic_name}")
                        st.markdown(
                            f"**Summary:** {topic_summaries[topic_id]['description']}"
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                        # Get topic words and create word cloud
                        if topic_id in topic_model.get_topics():
                            words = dict(topic_model.get_topics()[topic_id])

                            # Create and display word cloud with better styling
                            wordcloud = WordCloud(
                                width=400,
                                height=300,
                                background_color="white",
                                colormap="viridis",
                                max_words=100,
                                contour_width=1,
                                contour_color="steelblue",
                                prefer_horizontal=1,
                            ).generate_from_frequencies(words)

                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.imshow(wordcloud, interpolation="bilinear")
                            ax.axis("off")
                            ax.set_title(
                                f"Key Terms in '{topic_name}'", fontsize=14, pad=20
                            )
                            plt.tight_layout()
                            st.pyplot(fig)

                with col2:
                    # Display posts for this topic
                    topic_posts = df[df["topic_name"] == topic_name].reset_index(
                        drop=True
                    )

                    if not topic_posts.empty:
                        st.markdown(f"#### Sample Posts ({len(topic_posts)} total)")

                        # Create a more visually appealing post display
                        for idx, row in topic_posts.head(5).iterrows():
                            with st.expander(f"📝 {row['title']}"):
                                st.markdown(
                                    f"""
                                    <div style="padding: 10px; border-radius: 5px; background-color: #f8f9fa;">
                                        {row['post text'] if row['post text'] and row['post text'] != 'nan' else '<em>No post text</em>'}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                if row["comments"]:
                                    st.markdown("##### 💬 Sample Comments:")
                                    for c_idx, comment in enumerate(
                                        row["comments"][:3], 1
                                    ):
                                        st.markdown(
                                            f"""
                                            <div style="padding: 8px; border-left: 3px solid #FF4500; margin-bottom: 8px; background-color: #fafafa;">
                                                {comment}
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

        # Raw data section with improved styling
        with st.expander("View Raw Data"):
            # Add a search filter
            search_term = st.text_input("Search in titles", "")

            filtered_df = df
            if search_term:
                filtered_df = df[df["title"].str.contains(search_term, case=False)]

            # Add styling to the dataframe
            st.dataframe(
                filtered_df[["title", "topic", "topic_name"]],
                use_container_width=True,
                height=400,
            )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your inputs and try again.")

# Add footer with improved styling
st.markdown("---")
