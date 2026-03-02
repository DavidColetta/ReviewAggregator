"""
Single Business view — data loading, quality report, pipeline, and all four tabs.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pipeline import (
    run_pipeline, make_sample_data, THEME_KEYWORDS,
    compute_elbow_data, compute_data_quality, export_results_csv,
    load_data, extract_signals,
)
from views.components import sentiment_color, cluster_card, COLORS
from utils.charts import (
    bubble_chart, sentiment_donut, elbow_curve,
    silhouette_bars, rating_histogram,
)


# ─────────────────────────────────────────────
# Cached pipeline functions
# ─────────────────────────────────────────────

@st.cache_data
def get_results(df_json: str, n_clusters: int):
    df = pd.read_json(df_json)
    progress_bar = st.progress(0, text="Loading sentiment model...")

    def update_progress(current, total):
        pct = int(current / total * 100)
        progress_bar.progress(pct, text=f"Analyzing sentiment... {current}/{total} reviews")

    df_out, summaries, pca_meta = run_pipeline(
        df, n_clusters=n_clusters, progress_callback=update_progress
    )
    progress_bar.empty()
    return df_out, summaries, pca_meta


@st.cache_data
def get_signals_only(df_json: str) -> str:
    df = pd.read_json(df_json)
    df = load_data(df)
    df = extract_signals(df)
    return df.to_json()


@st.cache_data
def get_elbow_data(processed_df_json: str) -> dict:
    df = pd.read_json(processed_df_json)
    return compute_elbow_data(df)


@st.cache_data
def scrape_data(url: str, max_pages: int):
    from scraper import scrape_trustpilot
    return scrape_trustpilot(url, max_pages=max_pages)


# ─────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────

def render(data_source: str, business_name: str, trustpilot_url: str,
           uploaded_file, max_pages: int, n_clusters: int):
    """Entry point called from app.py."""

    # ── Load raw data ──
    raw_df = None

    if data_source == "Sample data":
        raw_df = make_sample_data()

    elif data_source == "Scrape Trustpilot":
        if not trustpilot_url:
            st.info("👈 Enter a Trustpilot URL or slug in the sidebar to get started.")
            return
        with st.spinner(f"Scraping Trustpilot ({max_pages} pages max)..."):
            try:
                raw_df, scraped_name = scrape_data(trustpilot_url, max_pages)
                business_name = scraped_name
                st.success(f"Scraped **{len(raw_df)} reviews** for **{scraped_name}**")
            except ValueError as e:
                st.error(f"Scraping failed: {e}")
                return

    elif data_source == "Upload CSV":
        if not uploaded_file:
            st.info("👈 Upload a CSV file in the sidebar to get started.")
            return
        raw_df = pd.read_csv(uploaded_file)

    # ── Header ──
    st.markdown(f"# {business_name}")
    st.caption("Customer review analysis — powered by clustering & sentiment")

    # ── Data quality report ──
    _render_quality_report(raw_df)

    # ── Run pipeline ──
    with st.spinner("Extracting signals from reviews..."):
        processed_json = get_signals_only(raw_df.to_json())

    elbow_result = get_elbow_data(processed_json)
    optimal_k = elbow_result["suggested_k"]

    auto_mode = not st.session_state.get("k_override", False)
    final_k = optimal_k if auto_mode else n_clusters

    with st.spinner(f"Clustering into {final_k} themes..."):
        df, summaries, pca_meta = get_results(raw_df.to_json(), final_k)

    if auto_mode:
        st.caption(f"🎯 Auto-selected **K = {final_k}** via elbow analysis. Move the sidebar slider to override.")
    else:
        st.caption(f"🎯 Using **K = {final_k}** — set manually. Elbow analysis recommends K = {optimal_k}.")

    # ── Sentiment blend toggle ──
    has_ratings = "rating" in df.columns and df["rating"].notna().any()
    if has_ratings:
        use_blended = st.toggle(
            "⭐ Blend star rating into sentiment score",
            value=True,
            key="use_blended_sentiment",
            help=(
                "When ON, the sentiment score is a 50/50 weighted average of the DistilBERT NLP "
                "score and the star rating normalized to −1…+1 (1★=−1, 3★=0, 5★=+1). "
                "When OFF, only the NLP model score is used."
            ),
        )
        if use_blended:
            st.caption(
                "⭐ **Blended mode active** — sentiment = 50% DistilBERT NLP + 50% star rating "
                "(1★ → −1.0 · 3★ → 0.0 · 5★ → +1.0)"
            )
    else:
        use_blended = False

    score_col = "sentiment_score_blended" if use_blended else "sentiment_score"
    label_col = "sentiment_label_blended" if use_blended else "sentiment_label"

    # ── Top metrics ──
    _render_metrics(df, business_name, summaries, score_col=score_col, label_col=label_col, use_blended=use_blended)

    st.markdown("---")

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📦 Theme Clusters", "🗺️ Review Map",
        "📈 Sentiment Breakdown", "🔬 Pipeline Analysis"
    ])

    with tab1:
        _tab_clusters(summaries, use_blended=use_blended)

    with tab2:
        _tab_review_map(df, summaries, pca_meta)

    with tab3:
        _tab_sentiment(df, summaries, score_col=score_col, label_col=label_col, use_blended=use_blended)

    with tab4:
        _tab_pipeline(elbow_result, pca_meta)


# ─────────────────────────────────────────────
# Sub-sections
# ─────────────────────────────────────────────

def _render_quality_report(raw_df: pd.DataFrame):
    quality = compute_data_quality(raw_df)
    with st.expander("📋 Data Quality Report", expanded=False):
        qcol1, qcol2, qcol3, qcol4 = st.columns(4)
        qcol1.metric("Raw reviews", f"{quality['total_raw']:,}")
        qcol2.metric(
            "After cleaning", f"{quality['total_clean']:,}",
            delta=f"-{quality['total_dropped']} dropped" if quality["total_dropped"] else "✓ none dropped",
            delta_color="off" if quality["total_dropped"] else "normal"
        )
        qcol3.metric("Avg review length", f"{quality['avg_length']} chars")
        qcol4.metric("Median review length", f"{quality['median_length']} chars")

        st.markdown("")
        dcol1, dcol2, dcol3 = st.columns(3)
        dcol1.metric("Very short (<50 chars)", quality["short_reviews"],
                     help="May not contain enough signal")
        dcol2.metric("Long (>500 chars)", quality["long_reviews"],
                     help="Rich in detail — good signal")
        if quality["has_ratings"] and quality["rating_stats"]:
            rs = quality["rating_stats"]
            dcol3.metric("Avg rating", f"{rs['avg_rating']} ★",
                         help=f"5★: {rs['pct_5_star']}%  |  1★: {rs['pct_1_star']}%")

        if quality.get("date_range"):
            st.caption(f"📅 {quality['date_range']['earliest']} → {quality['date_range']['latest']}")
        if quality["null_text"] > 0:
            st.warning(f"⚠️ {quality['null_text']} reviews had no text and were removed.")
        if quality["empty_text"] > 0:
            st.warning(f"⚠️ {quality['empty_text']} reviews were too short and were removed.")
        if quality["total_dropped"] == 0:
            st.success("✅ All reviews passed quality checks.")


def _render_metrics(
    df: pd.DataFrame, business_name: str, summaries: list,
    score_col: str = "sentiment_score",
    label_col: str = "sentiment_label",
    use_blended: bool = False,
):
    has_ratings = "rating" in df.columns and df["rating"].notna().any()
    effective_label = df[label_col] if label_col in df.columns else df["sentiment_label"]
    pct_positive = (effective_label == "Positive").mean() * 100
    pct_negative = (effective_label == "Negative").mean() * 100

    pos_label = "Positive ⭐" if use_blended else "Positive"
    neg_label = "Negative ⭐" if use_blended else "Negative"

    cols = st.columns([1, 1, 1, 1, 1.2])
    cols[0].metric("Total Reviews", f"{len(df):,}")
    cols[1].metric(pos_label, f"{pct_positive:.0f}%")
    cols[2].metric(neg_label, f"{pct_negative:.0f}%")
    if has_ratings:
        cols[3].metric("Avg Star Rating", f"{df['rating'].mean():.1f} ★")

    csv_data = export_results_csv(df, summaries)
    cols[4].download_button(
        label="⬇️ Download Results CSV",
        data=csv_data,
        file_name=f"{business_name.lower().replace(' ', '_')}_analysis.csv",
        mime="text/csv",
    )


def _tab_clusters(summaries: list, use_blended: bool = False):
    st.subheader("What customers are talking about")
    st.caption(f"Reviews grouped into {len(summaries)} themes by topic and sentiment.")
    for s in sorted(summaries, key=lambda x: x["review_count"], reverse=True):
        # Pass a copy with the active sentiment score so cluster_card renders correctly
        s_display = s.copy()
        if use_blended and "avg_sentiment_blended" in s:
            s_display["avg_sentiment"] = s["avg_sentiment_blended"]
        if use_blended and "sentiment_counts_blended" in s:
            s_display["sentiment_counts"] = s["sentiment_counts_blended"]
        st.markdown(cluster_card(s_display), unsafe_allow_html=True)


def _tab_review_map(df: pd.DataFrame, summaries: list, pca_meta: dict):
    st.subheader("Review Map")
    st.caption("Each bubble is a theme cluster. Size = number of reviews. Hover to read sample reviews.")

    id_to_name = {s["cluster_id"]: s["name"] for s in summaries}
    df["theme"] = df["cluster"].map(id_to_name)
    summary_lookup = {s["cluster_id"]: s for s in summaries}

    cluster_plot_df = df.groupby("cluster").agg(
        pca_x=("pca_x", "mean"),
        pca_y=("pca_y", "mean"),
    ).reset_index()
    cluster_plot_df["theme"]        = cluster_plot_df["cluster"].map(id_to_name)
    cluster_plot_df["review_count"] = cluster_plot_df["cluster"].map(lambda c: summary_lookup[c]["review_count"])
    cluster_plot_df["avg_sentiment"]= cluster_plot_df["cluster"].map(lambda c: summary_lookup[c]["avg_sentiment"])
    cluster_plot_df["top_words"]    = cluster_plot_df["cluster"].map(lambda c: ", ".join(summary_lookup[c]["top_words"][:5]))

    hover_texts = []
    for _, row in cluster_plot_df.iterrows():
        s = summary_lookup[row["cluster"]]
        reviews_text = "<br>".join(
            f'• {r[:80]}{"…" if len(r) > 80 else ""}' for r in s["sample_reviews"][:4]
        )
        hover_texts.append(
            f"<b>{row['theme']}</b><br>"
            f"{row['review_count']} reviews · sentiment {row['avg_sentiment']:+.2f}<br>"
            f"<i>Keywords: {row['top_words']}</i><br><br>"
            f"{reviews_text}"
        )

    fig = bubble_chart(cluster_plot_df, COLORS, hover_texts, pca_meta=pca_meta)
    st.plotly_chart(fig, use_container_width=True)

    col_v1, col_v2, col_v3 = st.columns(3)
    col_v1.metric("Variance explained (2D)", f"{pca_meta['variance_2d']:.0%}",
                  help="How much of the data variation is visible in this chart")
    col_v2.metric(f"Variance for clustering ({pca_meta['n_cluster_components']} components)",
                  f"{pca_meta['variance_cluster']:.0%}",
                  help="How much information was retained when clustering.")
    col_v3.metric("X / Y axis split",
                  f"{pca_meta['variance_x']:.0%} / {pca_meta['variance_y']:.0%}",
                  help="How much variance each axis captures individually")


def _tab_sentiment(
    df: pd.DataFrame, summaries: list,
    score_col: str = "sentiment_score",
    label_col: str = "sentiment_label",
    use_blended: bool = False,
):
    has_ratings = "rating" in df.columns and df["rating"].notna().any()
    col_a, col_b = st.columns(2)

    with col_a:
        title = "Overall Sentiment ⭐" if use_blended else "Overall Sentiment"
        st.subheader(title)
        active_label = df[label_col] if label_col in df.columns else df["sentiment_label"]
        sent_counts = active_label.value_counts().to_dict()
        st.plotly_chart(sentiment_donut(sent_counts), use_container_width=True)

    with col_b:
        if has_ratings:
            st.subheader("Rating Distribution")
            st.plotly_chart(rating_histogram(df["rating"]), use_container_width=True)
        else:
            active_score = score_col if score_col in df.columns else "sentiment_score"
            dist_title = "Blended Sentiment Score Distribution" if use_blended else "Sentiment Score Distribution"
            st.subheader(dist_title)
            fig_hist = px.histogram(
                df, x=active_score, nbins=20,
                color_discrete_sequence=["#2980b9"],
                template="plotly_white",
                labels={active_score: "Sentiment Score"},
            )
            fig_hist.update_layout(height=320, paper_bgcolor="#faf8f5",
                                   plot_bgcolor="#faf8f5", margin=dict(t=10))
            st.plotly_chart(fig_hist, use_container_width=True)

    sentiment_col_label = "Sentiment ⭐" if use_blended else "Sentiment"
    st.subheader("Theme Summary")
    table_rows = []
    for s in sorted(summaries, key=lambda x: x["review_count"], reverse=True):
        active_sent = s.get("avg_sentiment_blended", s["avg_sentiment"]) if use_blended else s["avg_sentiment"]
        emoji = "🟢" if active_sent >= 0.05 else ("🔴" if active_sent <= -0.05 else "🟡")
        table_rows.append({
            "Theme":              s["name"],
            "Reviews":            s["review_count"],
            "Avg Rating":         f"{s['avg_rating']} ★" if s["avg_rating"] else "—",
            sentiment_col_label:  f"{emoji} {active_sent:+.2f}",
            "Top Keywords":       ", ".join(s["top_words"][:4]),
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


def _tab_pipeline(elbow_result: dict, pca_meta: dict):
    st.subheader("Optimal Number of Clusters")
    st.caption("Computed by running K-Means for K=2 to K=8 and measuring cluster quality.")

    rec_col1, rec_col2 = st.columns(2)
    rec_col1.success(
        f"✅ **Using K = {elbow_result['elbow_k']}** (elbow method)\n\n"
        f"The point where adding more clusters gives diminishing returns."
    )
    rec_col2.info(
        f"📊 **Best silhouette score at K = {elbow_result['silhouette_k']}**\n\n"
        f"Measures how well-separated clusters are from each other."
    )

    ecol1, ecol2 = st.columns(2)
    with ecol1:
        fig = elbow_curve(elbow_result["k_values"], elbow_result["inertias"], elbow_result["elbow_k"])
        fig.update_layout(title="Inertia vs K (Elbow Curve)",
                          yaxis_title="Inertia (lower = tighter clusters)")
        st.plotly_chart(fig, use_container_width=True)

    with ecol2:
        fig = silhouette_bars(elbow_result["k_values"], elbow_result["silhouettes"],
                              elbow_result["silhouette_k"])
        fig.update_layout(title="Silhouette Score vs K (higher = better)")
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "**Inertia** measures how tightly packed reviews are within each cluster. "
        "**Silhouette score** measures how well-separated clusters are — higher is better."
    )

    st.subheader("PCA Variance Breakdown")
    st.caption(
        f"K-Means ran on {pca_meta['n_cluster_components']} PCA components "
        f"capturing {pca_meta['variance_cluster']:.0%} of total variance. "
        f"The 2D chart shows {pca_meta['variance_2d']:.0%}."
    )
    variance_df = pd.DataFrame({
        "Component": ["Component 1 (X axis)", "Component 2 (Y axis)", "Components 3+", "Total (clustering)"],
        "Variance Explained": [
            f"{pca_meta['variance_x']:.1%}", f"{pca_meta['variance_y']:.1%}",
            f"{pca_meta['variance_cluster'] - pca_meta['variance_2d']:.1%}",
            f"{pca_meta['variance_cluster']:.1%}",
        ],
        "Used for": ["Visualization", "Visualization", "Clustering only", "Clustering"],
    })
    st.dataframe(variance_df, use_container_width=True, hide_index=True)