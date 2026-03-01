import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pipeline import run_pipeline, make_sample_data, THEME_KEYWORDS

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Review Aggregator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS — warm editorial style
# ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Sans 3', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
    }
    .main { background-color: #faf8f5; }
    .block-container { padding-top: 2rem; }

    /* Cluster cards */
    .cluster-card {
        background: white;
        border-left: 5px solid #c0392b;
        border-radius: 4px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .cluster-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.3rem;
    }
    .cluster-meta {
        font-size: 0.85rem;
        color: #888;
        margin-bottom: 0.6rem;
    }
    .tag {
        display: inline-block;
        background: #f0ebe3;
        color: #555;
        font-size: 0.75rem;
        padding: 2px 10px;
        border-radius: 20px;
        margin: 2px 3px 2px 0;
    }
    .review-quote {
        font-style: italic;
        color: #444;
        font-size: 0.9rem;
        border-left: 3px solid #e8e0d5;
        padding-left: 0.8rem;
        margin: 0.4rem 0;
    }
    .sentiment-pos { color: #27ae60; font-weight: 600; }
    .sentiment-neg { color: #c0392b; font-weight: 600; }
    .sentiment-neu { color: #888; font-weight: 600; }

    /* Metric override */
    [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', serif !important;
        font-size: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 Review Aggregator")
    st.caption("Understand what your customers are really saying.")
    st.markdown("---")

    business_name = st.text_input("Business name", value="Luigi's Bistro")
    n_clusters = st.slider("Number of themes to find", min_value=2, max_value=8, value=5)

    st.markdown("---")
    st.markdown("**Data source**")
    use_sample = st.checkbox("Use sample data", value=True)
    uploaded_file = None
    if not use_sample:
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            help="Required column: review_text\nOptional: rating"
        )
        st.caption("Required column: `review_text`\nOptional: `rating`")


# ─────────────────────────────────────────────
# Load & run pipeline
# ─────────────────────────────────────────────

@st.cache_data
def get_results(df_json: str, n_clusters: int):
    df = pd.read_json(df_json)
    return run_pipeline(df, n_clusters=n_clusters)

def sentiment_color(score):
    if score >= 0.05:
        return "sentiment-pos", "● Positive"
    elif score <= -0.05:
        return "sentiment-neg", "● Negative"
    return "sentiment-neu", "● Neutral"

# Decide data source
raw_df = None
if use_sample:
    raw_df = make_sample_data()
elif uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────

st.markdown(f"# {business_name}")
st.caption("Customer review analysis — powered by clustering & sentiment")

if raw_df is None:
    st.info("👈 Check **Use sample data** in the sidebar, or upload your own CSV to get started.")
    st.stop()

with st.spinner("Analyzing reviews..."):
    df, summaries = get_results(raw_df.to_json(), n_clusters)

# ─────────────────────────────────────────────
# Top-level metrics
# ─────────────────────────────────────────────

total_reviews = len(df)
avg_sentiment = df["sentiment_score"].mean()
pct_positive = (df["sentiment_label"] == "Positive").mean() * 100
pct_negative = (df["sentiment_label"] == "Negative").mean() * 100
has_ratings = "rating" in df.columns and df["rating"].notna().any()

cols = st.columns(4 if has_ratings else 3)
cols[0].metric("Total Reviews", f"{total_reviews:,}")
cols[1].metric("Positive", f"{pct_positive:.0f}%")
cols[2].metric("Negative", f"{pct_negative:.0f}%")
if has_ratings:
    cols[3].metric("Avg Star Rating", f"{df['rating'].mean():.1f} ★")

st.markdown("---")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📦 Theme Clusters", "🗺️ Review Map", "📈 Sentiment Breakdown"])

# ════════════════════════════════════════════
# TAB 1: Cluster cards
# ════════════════════════════════════════════

with tab1:
    st.subheader("What customers are talking about")
    st.caption(f"Reviews grouped into {len(summaries)} themes by topic and sentiment.")

    for s in sorted(summaries, key=lambda x: x["review_count"], reverse=True):
        sent_class, sent_text = sentiment_color(s["avg_sentiment"])
        rating_str = f" · {s['avg_rating']} ★" if s["avg_rating"] else ""

        # Build tag pills from top words
        tags_html = "".join(f'<span class="tag">{w}</span>' for w in s["top_words"])

        # Pick 3 sample reviews (mix of pos and neg)
        quotes_html = "".join(
            f'<div class="review-quote">"{r[:160]}{"..." if len(r) > 160 else ""}"</div>'
            for r in s["sample_reviews"][:3]
        )

        card_html = f"""
        <div class="cluster-card">
            <div class="cluster-title">{s['name']}</div>
            <div class="cluster-meta">
                {s['review_count']} reviews
                <span class="{sent_class}"> · {sent_text}</span>
                {rating_str}
            </div>
            <div style="margin-bottom:0.7rem">{tags_html}</div>
            {quotes_html}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 2: PCA scatter map
# ════════════════════════════════════════════

with tab2:
    st.subheader("Review Map")
    st.caption("Each dot is a review. Reviews that cluster together share similar language and themes. Hover to read.")

    # Map cluster ids to names
    id_to_name = {s["cluster_id"]: s["name"] for s in summaries}
    df["theme"] = df["cluster"].map(id_to_name)

    fig = px.scatter(
        df,
        x="pca_x",
        y="pca_y",
        color="theme",
        hover_data={"review_text": True, "sentiment_score": ":.2f",
                    "rating": True if has_ratings else False,
                    "pca_x": False, "pca_y": False},
        custom_data=["review_text", "sentiment_score"],
        color_discrete_sequence=["#c0392b", "#e67e22", "#27ae60", "#2980b9", "#8e44ad", "#16a085", "#d35400"],
        template="plotly_white",
        labels={"pca_x": "", "pca_y": "", "theme": "Theme"},
    )

    fig.update_traces(
        marker=dict(size=9, opacity=0.75, line=dict(width=0.5, color="white")),
        hovertemplate="<b>%{customdata[0]}</b><br>Sentiment: %{customdata[1]:.2f}<extra></extra>"
    )
    fig.update_layout(
        height=520,
        font_family="Source Sans 3",
        legend_title_text="Theme",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#faf8f5",
        paper_bgcolor="#faf8f5",
    )
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════
# TAB 3: Sentiment breakdown
# ════════════════════════════════════════════

with tab3:
    col_a, col_b = st.columns(2)

    # Overall sentiment donut
    with col_a:
        st.subheader("Overall Sentiment")
        sent_counts = df["sentiment_label"].value_counts()
        colors = {"Positive": "#27ae60", "Neutral": "#bdc3c7", "Negative": "#c0392b"}
        fig_donut = go.Figure(data=[go.Pie(
            labels=sent_counts.index,
            values=sent_counts.values,
            hole=0.55,
            marker_colors=[colors.get(l, "#aaa") for l in sent_counts.index],
            textfont_size=13,
        )])
        fig_donut.update_layout(
            height=320,
            showlegend=True,
            font_family="Source Sans 3",
            paper_bgcolor="#faf8f5",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Sentiment per theme bar
    with col_b:
        st.subheader("Sentiment by Theme")
        theme_sent = df.groupby("theme")["sentiment_score"].mean().sort_values()
        bar_colors = ["#c0392b" if v < 0 else "#27ae60" for v in theme_sent.values]
        fig_bar = go.Figure(go.Bar(
            x=theme_sent.values,
            y=theme_sent.index,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:+.2f}" for v in theme_sent.values],
            textposition="outside",
        ))
        fig_bar.update_layout(
            height=320,
            xaxis_title="Avg Sentiment Score",
            yaxis_title="",
            font_family="Source Sans 3",
            paper_bgcolor="#faf8f5",
            plot_bgcolor="#faf8f5",
            margin=dict(l=10, r=40, t=10, b=10),
            xaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinecolor="#ddd"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Cluster summary table
    st.subheader("Theme Summary")

    table_rows = []
    for s in sorted(summaries, key=lambda x: x["review_count"], reverse=True):
        sent_emoji = "🟢" if s["avg_sentiment"] >= 0.05 else ("🔴" if s["avg_sentiment"] <= -0.05 else "🟡")
        table_rows.append({
            "Theme":           s["name"],
            "Reviews":         s["review_count"],
            "Avg Rating":      f"{s['avg_rating']} ★" if s["avg_rating"] else "—",
            "Sentiment":       f"{sent_emoji} {s['avg_sentiment']:+.2f}",
            "Top Keywords":    ", ".join(s["top_words"][:4]),
        })

    summary_df = pd.DataFrame(table_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("Review Aggregator · Built with Streamlit, scikit-learn & VADER")