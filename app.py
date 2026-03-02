import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pipeline import run_pipeline, make_sample_data, THEME_KEYWORDS, compute_elbow_data, compute_data_quality, export_results_csv

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

    # Use session_state to detect if user has manually touched the slider
    if "k_override" not in st.session_state:
        st.session_state.k_override = False

    def on_slider_change():
        st.session_state.k_override = True

    n_clusters = st.slider(
        "Number of themes",
        min_value=2, max_value=8, value=5,
        key="n_clusters_slider",
        on_change=on_slider_change,
        help="Auto-selects optimal K via elbow analysis until you move this slider."
    )

    if st.session_state.k_override:
        if st.button("↩️ Reset to auto", use_container_width=True):
            st.session_state.k_override = False
            # Button click triggers rerun automatically — no need for st.rerun()

    st.markdown("---")
    st.markdown("**Data source**")
    data_source = st.radio(
        "Choose source",
        ["Sample data", "Scrape Trustpilot", "Upload CSV"],
        label_visibility="collapsed",
        key="data_source_radio",
    )

    trustpilot_url = None
    uploaded_file = None
    max_pages = 5
    business_name = "Luigi's Bistro"

    if data_source == "Sample data":
        business_name = st.text_input("Business name", value="Luigi's Bistro", key="sample_business_name")

    elif data_source == "Scrape Trustpilot":
        trustpilot_url = st.text_input(
            "Trustpilot URL or slug",
            placeholder="e.g. dominos.com",
            help="Paste a full URL like https://www.trustpilot.com/review/dominos.com or just the slug",
            key="trustpilot_url_input",
        )
        max_pages = st.slider("Max pages to scrape", 1, 20, 5,
                              help="Each page has ~20 reviews. More pages = slower but richer analysis.",
                              key="max_pages_slider")
        st.caption("🕐 ~1 second per page")

    elif data_source == "Upload CSV":
        business_name = st.text_input("Business name", placeholder="e.g. Mario's Pizzeria", key="csv_business_name")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")
        st.caption("Required column: `review_text`\nOptional: `rating`")


# ─────────────────────────────────────────────
# Load & run pipeline
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Load & run pipeline
# ─────────────────────────────────────────────

@st.cache_data
def get_results(df_json: str, n_clusters: int):
    df = pd.read_json(df_json)

    progress_bar = st.progress(0, text="Loading sentiment model...")

    def update_progress(current, total):
        pct = int(current / total * 100)
        progress_bar.progress(pct, text=f"Analyzing sentiment... {current}/{total} reviews")

    df_out, summaries, pca_meta = run_pipeline(df, n_clusters=n_clusters, progress_callback=update_progress)
    progress_bar.empty()
    return df_out, summaries, pca_meta

@st.cache_data
def scrape_data(url: str, max_pages: int):
    from scraper import scrape_trustpilot
    return scrape_trustpilot(url, max_pages=max_pages)

def sentiment_color(score):
    if score >= 0.05:
        return "sentiment-pos", "● Positive"
    elif score <= -0.05:
        return "sentiment-neg", "● Negative"
    return "sentiment-neu", "● Neutral"

# ── Decide data source ──
raw_df = None

if data_source == "Sample data":
    raw_df = make_sample_data()

elif data_source == "Scrape Trustpilot":
    if trustpilot_url:
        with st.spinner(f"Scraping Trustpilot ({max_pages} pages max)..."):
            try:
                raw_df, scraped_name = scrape_data(trustpilot_url, max_pages)
                business_name = scraped_name  # override sidebar name with real name
                st.success(f"Scraped **{len(raw_df)} reviews** for **{scraped_name}**")
            except ValueError as e:
                st.error(f"Scraping failed: {e}")
                st.stop()
    else:
        st.info("👈 Enter a Trustpilot URL or slug in the sidebar to get started.")
        st.stop()

elif data_source == "Upload CSV":
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
    else:
        st.info("👈 Upload a CSV file in the sidebar to get started.")
        st.stop()

# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────

st.markdown(f"# {business_name}")
st.caption("Customer review analysis — powered by clustering & sentiment")

# ─────────────────────────────────────────────
# Data quality report
# ─────────────────────────────────────────────

quality = compute_data_quality(raw_df)

with st.expander("📋 Data Quality Report", expanded=False):
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    qcol1.metric("Raw reviews", f"{quality['total_raw']:,}")
    qcol2.metric("After cleaning", f"{quality['total_clean']:,}",
                 delta=f"-{quality['total_dropped']} dropped" if quality["total_dropped"] else "✓ none dropped",
                 delta_color="off" if quality["total_dropped"] else "normal")
    qcol3.metric("Avg review length", f"{quality['avg_length']} chars")
    qcol4.metric("Median review length", f"{quality['median_length']} chars")

    st.markdown("")
    dcol1, dcol2, dcol3 = st.columns(3)
    dcol1.metric("Very short reviews (<50 chars)", quality["short_reviews"],
                 help="These may not contain enough signal for analysis")
    dcol2.metric("Long reviews (>500 chars)", quality["long_reviews"],
                 help="Rich in detail — good signal for clustering")
    if quality["has_ratings"] and quality["rating_stats"]:
        rs = quality["rating_stats"]
        dcol3.metric("Avg rating", f"{rs['avg_rating']} ★",
                     help=f"5★: {rs['pct_5_star']}%  |  1★: {rs['pct_1_star']}%")

    if quality["date_range"]:
        st.caption(f"📅 Reviews span {quality['date_range']['earliest']} → {quality['date_range']['latest']}")

    if quality["null_text"] > 0:
        st.warning(f"⚠️ {quality['null_text']} reviews had no text and were removed.")
    if quality["empty_text"] > 0:
        st.warning(f"⚠️ {quality['empty_text']} reviews were too short (<10 chars) and were removed.")
    if quality["total_dropped"] == 0:
        st.success("✅ All reviews passed quality checks — no data dropped.")

# ─────────────────────────────────────────────
# Elbow analysis + main pipeline
# Strategy: run signals-only pass first (cached), use it for elbow,
# then run full pipeline once with the optimal K.
# ─────────────────────────────────────────────

@st.cache_data
def get_signals_only(df_json: str) -> str:
    """Run load + extract_signals only — no clustering. Returns processed df as JSON."""
    from pipeline import load_data, extract_signals
    df = pd.read_json(df_json)
    df = load_data(df)
    df = extract_signals(df)
    return df.to_json()

@st.cache_data
def get_elbow_data(processed_df_json: str) -> dict:
    """Run elbow analysis on already-processed df (theme cols present)."""
    df = pd.read_json(processed_df_json)
    return compute_elbow_data(df)

# Step 1: extract signals (fast after first run — cached)
with st.spinner("Extracting signals from reviews..."):
    processed_json = get_signals_only(raw_df.to_json())

# Step 2: find optimal K
elbow_result = get_elbow_data(processed_json)
optimal_k = elbow_result["suggested_k"]

# Step 3: let user override, but default to optimal
auto_mode = not st.session_state.get("k_override", False)
final_k = optimal_k if auto_mode else n_clusters

# Step 4: run full pipeline with final K
with st.spinner(f"Clustering into {final_k} themes..."):
    df, summaries, pca_meta = get_results(raw_df.to_json(), final_k)

if auto_mode:
    st.caption(f"🎯 Auto-selected **K = {final_k}** clusters via elbow analysis. Move the sidebar slider to override.")
else:
    st.caption(f"🎯 Using **K = {final_k}** clusters — set manually. Elbow analysis recommends K = {optimal_k}.")

# ─────────────────────────────────────────────
# Top-level metrics + download button
# ─────────────────────────────────────────────

total_reviews = len(df)
avg_sentiment = df["sentiment_score"].mean()
pct_positive = (df["sentiment_label"] == "Positive").mean() * 100
pct_negative = (df["sentiment_label"] == "Negative").mean() * 100
has_ratings = "rating" in df.columns and df["rating"].notna().any()

metric_cols = st.columns([1, 1, 1, 1, 1.2])
metric_cols[0].metric("Total Reviews", f"{total_reviews:,}")
metric_cols[1].metric("Positive", f"{pct_positive:.0f}%")
metric_cols[2].metric("Negative", f"{pct_negative:.0f}%")
if has_ratings:
    metric_cols[3].metric("Avg Star Rating", f"{df['rating'].mean():.1f} ★")

# CSV download button
csv_data = export_results_csv(df, summaries)
metric_cols[4].download_button(
    label="⬇️ Download Results CSV",
    data=csv_data,
    file_name=f"{business_name.lower().replace(' ', '_')}_review_analysis.csv",
    mime="text/csv",
)

st.markdown("---")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["📦 Theme Clusters", "🗺️ Review Map", "📈 Sentiment Breakdown", "🔬 Pipeline Analysis"])

# ════════════════════════════════════════════
# TAB 1: Cluster cards
# ════════════════════════════════════════════

with tab1:
    st.subheader("What customers are talking about")
    st.caption(f"Reviews grouped into {len(summaries)} themes by topic and sentiment.")

    for s in sorted(summaries, key=lambda x: x["review_count"], reverse=True):
        sent_class, sent_text = sentiment_color(s["avg_sentiment"])
        rating_str = f" · {s['avg_rating']} ★" if s["avg_rating"] else ""

        # Border color driven by sentiment_tag
        tag = s.get("sentiment_tag", "Mixed")
        if tag == "Praise":
            border_color = "#27ae60"
        elif tag == "Complaints":
            border_color = "#c0392b"
        else:
            border_color = "#f39c12"

        # Build tag pills from top words
        tags_html = "".join(f'<span class="tag">{w}</span>' for w in s["top_words"])

        # Pick 3 sample reviews (mix of pos and neg)
        quotes_html = "".join(
            f'<div class="review-quote">"{r[:160]}{"..." if len(r) > 160 else ""}"</div>'
            for r in s["sample_reviews"][:3]
        )

        card_html = f"""
        <div class="cluster-card" style="border-left-color: {border_color}">
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
    st.caption("Each bubble is a theme cluster. Size = number of reviews. Hover to read sample reviews.")

    # Map cluster ids to names
    id_to_name = {s["cluster_id"]: s["name"] for s in summaries}
    df["theme"] = df["cluster"].map(id_to_name)

    # Build one row per cluster using centroid of PCA coords
    cluster_plot_df = df.groupby("cluster").agg(
        pca_x=("pca_x", "mean"),
        pca_y=("pca_y", "mean"),
    ).reset_index()
    cluster_plot_df["theme"] = cluster_plot_df["cluster"].map(id_to_name)

    # Merge in summary stats
    summary_lookup = {s["cluster_id"]: s for s in summaries}
    cluster_plot_df["review_count"]  = cluster_plot_df["cluster"].map(lambda c: summary_lookup[c]["review_count"])
    cluster_plot_df["avg_sentiment"] = cluster_plot_df["cluster"].map(lambda c: summary_lookup[c]["avg_sentiment"])
    cluster_plot_df["avg_rating"]    = cluster_plot_df["cluster"].map(lambda c: summary_lookup[c]["avg_rating"] or "N/A")
    cluster_plot_df["top_words"]     = cluster_plot_df["cluster"].map(lambda c: ", ".join(summary_lookup[c]["top_words"][:5]))

    # Build hover text — include sample reviews
    def build_hover(row):
        s = summary_lookup[row["cluster"]]
        reviews_text = "<br>".join(
            f'• {r[:80]}{"…" if len(r) > 80 else ""}'
            for r in s["sample_reviews"][:4]
        )
        return (
            f"<b>{row['theme']}</b><br>"
            f"{row['review_count']} reviews · sentiment {row['avg_sentiment']:+.2f}<br>"
            f"<i>Keywords: {row['top_words']}</i><br><br>"
            f"{reviews_text}"
        )

    cluster_plot_df["hover_text"] = cluster_plot_df.apply(build_hover, axis=1)

    colors = ["#c0392b", "#e67e22", "#27ae60", "#2980b9", "#8e44ad", "#16a085", "#d35400", "#2c3e50"]

    fig = go.Figure()
    for i, row in cluster_plot_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["pca_x"]],
            y=[row["pca_y"]],
            mode="markers+text",
            marker=dict(
                size=row["review_count"] * 6,  # scale bubble by review count
                sizemode="area",
                sizeref=2.0 * max(cluster_plot_df["review_count"]) / (80 ** 2),
                color=colors[i % len(colors)],
                opacity=0.7,
                line=dict(width=2, color="white"),
            ),
            text=row["theme"].split(" — ")[0],  # short label on bubble
            textposition="middle center",
            textfont=dict(size=13, color="black", family="Source Sans 3"),
            hovertemplate=row["hover_text"] + "<extra></extra>",
            name=row["theme"],
        ))

    fig.update_layout(
        height=620,
        font_family="Source Sans 3",
        showlegend=False,
        xaxis=dict(
            title=dict(text=pca_meta["axis_x_label"], font=dict(size=11, color="#888")),
            showgrid=False, zeroline=False, showticklabels=False,
        ),
        yaxis=dict(
            title=dict(text=pca_meta["axis_y_label"], font=dict(size=11, color="#888")),
            showgrid=False, zeroline=False, showticklabels=False,
        ),
        plot_bgcolor="#faf8f5",
        paper_bgcolor="#faf8f5",
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Source Sans 3",
            bordercolor="#ddd",
        ),
        margin=dict(t=20, b=60, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    col_v1, col_v2, col_v3 = st.columns(3)
    col_v1.metric(
        "Variance explained (2D view)",
        f"{pca_meta['variance_2d']:.0%}",
        help="How much of the data variation is visible in this chart"
    )
    col_v2.metric(
        f"Variance for clustering ({pca_meta['n_cluster_components']} components)",
        f"{pca_meta['variance_cluster']:.0%}",
        help="How much information was retained when clustering. Higher = better clusters."
    )
    col_v3.metric(
        "X / Y axis split",
        f"{pca_meta['variance_x']:.0%} / {pca_meta['variance_y']:.0%}",
        help="How much variance each axis captures individually"
    )


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

    # Star rating distribution (if available)
    with col_b:
        if has_ratings:
            st.subheader("Rating Distribution")
            rating_counts = df["rating"].value_counts().sort_index()
            fig_ratings = go.Figure(go.Bar(
                x=rating_counts.index,
                y=rating_counts.values,
                marker_color=["#c0392b", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"],
                text=rating_counts.values,
                textposition="outside",
            ))
            fig_ratings.update_layout(
                height=320,
                xaxis_title="Stars",
                yaxis_title="Reviews",
                font_family="Source Sans 3",
                paper_bgcolor="#faf8f5",
                plot_bgcolor="#faf8f5",
                margin=dict(t=10),
                xaxis=dict(tickmode="linear"),
            )
            st.plotly_chart(fig_ratings, use_container_width=True)
        else:
            st.subheader("Sentiment Score Distribution")
            fig_hist = px.histogram(
                df, x="sentiment_score", nbins=20,
                color_discrete_sequence=["#2980b9"],
                template="plotly_white",
                labels={"sentiment_score": "Sentiment Score"},
            )
            fig_hist.update_layout(
                height=320,
                font_family="Source Sans 3",
                paper_bgcolor="#faf8f5",
                plot_bgcolor="#faf8f5",
                margin=dict(t=10),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # Theme summary table
    st.subheader("Theme Summary")
    table_rows = []
    for s in sorted(summaries, key=lambda x: x["review_count"], reverse=True):
        sent_emoji = "🟢" if s["avg_sentiment"] >= 0.05 else ("🔴" if s["avg_sentiment"] <= -0.05 else "🟡")
        table_rows.append({
            "Theme":        s["name"],
            "Reviews":      s["review_count"],
            "Avg Rating":   f"{s['avg_rating']} ★" if s["avg_rating"] else "—",
            "Sentiment":    f"{sent_emoji} {s['avg_sentiment']:+.2f}",
            "Top Keywords": ", ".join(s["top_words"][:4]),
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════
# TAB 4: Pipeline Analysis
# ════════════════════════════════════════════

with tab4:
    st.subheader("Optimal Number of Clusters")
    st.caption("Computed by running K-Means for K=2 to K=8 and measuring cluster quality.")

    elbow = elbow_result  # already computed above — reuse cached result

    # Recommended K callout
    rec_col1, rec_col2 = st.columns(2)
    rec_col1.success(f"✅ **Using K = {elbow['elbow_k']}** (elbow method)  \n"
                     f"The point where adding more clusters gives diminishing returns.")
    rec_col2.info(f"📊 **Best silhouette score at K = {elbow['silhouette_k']}**  \n"
                  f"Measures how well-separated clusters are from each other.")

    ecol1, ecol2 = st.columns(2)

    # Elbow curve — inertia
    with ecol1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=elbow["k_values"], y=elbow["inertias"],
            mode="lines+markers",
            line=dict(color="#2980b9", width=2),
            marker=dict(size=8, color="#2980b9"),
            name="Inertia",
        ))
        # Highlight elbow K
        elbow_idx = elbow["k_values"].index(elbow["elbow_k"])
        fig_elbow.add_trace(go.Scatter(
            x=[elbow["elbow_k"]], y=[elbow["inertias"][elbow_idx]],
            mode="markers", marker=dict(size=14, color="#c0392b", symbol="star"),
            name=f"Elbow (K={elbow['elbow_k']})",
        ))
        fig_elbow.update_layout(
            title="Inertia vs K (Elbow Curve)",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Inertia (lower = tighter clusters)",
            height=320, template="plotly_white",
            font_family="Source Sans 3", paper_bgcolor="#faf8f5",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    # Silhouette score chart
    with ecol2:
        sil_colors = [
            "#27ae60" if k == elbow["silhouette_k"] else "#bdc3c7"
            for k in elbow["k_values"]
        ]
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Bar(
            x=elbow["k_values"], y=elbow["silhouettes"],
            marker_color=sil_colors,
            text=[f"{s:.3f}" for s in elbow["silhouettes"]],
            textposition="outside",
            name="Silhouette Score",
        ))
        fig_sil.update_layout(
            title="Silhouette Score vs K (higher = better)",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Silhouette Score",
            height=320, template="plotly_white",
            font_family="Source Sans 3", paper_bgcolor="#faf8f5",
            yaxis=dict(range=[0, max(elbow["silhouettes"]) * 1.2]),
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    st.caption(
        "**Inertia** measures how tightly packed reviews are within each cluster — lower is better, "
        "but always decreases as K increases. The **elbow point** is where the curve bends. "
        "**Silhouette score** measures how well-separated clusters are from each other — higher is better."
    )

    # PCA variance breakdown table
    st.subheader("PCA Variance Breakdown")
    st.caption(f"K-Means ran on {pca_meta['n_cluster_components']} PCA components "
               f"capturing {pca_meta['variance_cluster']:.0%} of total variance. "
               f"The 2D chart shows {pca_meta['variance_2d']:.0%}.")

    variance_data = pd.DataFrame({
        "Component":          ["Component 1 (X axis)", "Component 2 (Y axis)", "Components 3+", "Total (clustering)"],
        "Variance Explained": [
            f"{pca_meta['variance_x']:.1%}",
            f"{pca_meta['variance_y']:.1%}",
            f"{pca_meta['variance_cluster'] - pca_meta['variance_2d']:.1%}",
            f"{pca_meta['variance_cluster']:.1%}",
        ],
        "Used for":           ["Visualization", "Visualization", "Clustering only", "Clustering"],
    })
    st.dataframe(variance_data, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("Review Aggregator · Built with Streamlit, scikit-learn & DistilBERT")