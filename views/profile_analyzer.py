# [NEW FILE] views/profile_analyzer.py (FIXED)
"""
Streamlit UI for Reviewer Profile Analysis
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from profile_analyzer import analyze_reviewers, cluster_reviewers, name_reviewer_cluster
from pipeline import THEME_KEYWORDS
from views.components import COLORS


def render_profile_analysis(df: pd.DataFrame):
    """
    Main render function for the profile analysis tab.
    """
    st.subheader("👤 Reviewer Profile Analysis")
    st.caption("Understand who your reviewers are - their patterns, preferences, and behaviors")
    
    if len(df) < 5:
        st.warning("Need at least 5 reviews to analyze reviewer profiles.")
        return
    
    with st.spinner("Analyzing reviewer profiles..."):
        # Get reviewer profiles
        profiles_df = analyze_reviewers(df)
        
        # Cluster reviewers by behavior
        n_clusters = st.slider(
            "Number of reviewer types to identify",
            min_value=2, max_value=6, value=4,
            key="profile_n_clusters",
            help="Group similar reviewers into these many behavioral types"
        )
        profiles_df = cluster_reviewers(profiles_df, n_clusters=n_clusters)
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Unique Reviewers", len(profiles_df))
    col2.metric("Avg Reviews/Reviewer", f"{df.shape[0] / len(profiles_df):.1f}")
    col3.metric("Most Active", f"{profiles_df['total_reviews'].max()} reviews")
    col4.metric("Reviewer Types", profiles_df['reviewer_cluster'].nunique() if 'reviewer_cluster' in profiles_df.columns else 1)
    
    # Tabs for different profile views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Reviewer Map", "📊 Reviewer Types", 
        "🏆 Top Reviewers", "📈 Profile Details"
    ])
    
    with tab1:
        render_reviewer_map(profiles_df, original_df=df)
    
    with tab2:
        render_reviewer_clusters(profiles_df)
    
    with tab3:
        render_top_reviewers(profiles_df)
    
    with tab4:
        render_profile_details(profiles_df, df)
    
    # Download profiles
    # Create a copy without the sample_reviews column for CSV export
    export_df = profiles_df.copy()
    if "sample_reviews" in export_df.columns:
        export_df = export_df.drop(columns=["sample_reviews"])
    csv_data = export_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Reviewer Profiles CSV",
        data=csv_data,
        file_name="reviewer_profiles.csv",
        mime="text/csv",
    )

def render_reviewer_map(profiles_df: pd.DataFrame, original_df: pd.DataFrame = None):
    """
    Render a scatter plot of reviewers in PCA space.
    Now includes sample reviews in hover tooltips.
    
    Args:
        profiles_df: DataFrame with reviewer profiles
        original_df: Original reviews DataFrame to pull sample reviews from
    """
    st.subheader("Reviewer Behavior Map")
    st.caption(
        "Each dot is a reviewer. Position reflects their review style, topic preferences, "
        "and sentiment patterns. Size = number of reviews. **Hover over any dot to see their reviews!**"
    )
    
    if "reviewer_pca_x" not in profiles_df.columns:
        st.info("Not enough data for PCA visualization.")
        return
    
    fig = go.Figure()
    
    # Prepare hover data for each reviewer
    hover_texts = []
    custom_data = []
    
    for _, reviewer in profiles_df.iterrows():
        reviewer_name = reviewer["reviewer_name"]
        
        # Get sample reviews from this reviewer
        sample_reviews_text = ""
        if original_df is not None and "reviewer_name" in original_df.columns:
            reviewer_reviews = original_df[original_df["reviewer_name"] == reviewer_name]
            if not reviewer_reviews.empty:
                # Get 3 sample reviews with sentiment
                samples = []
                for _, review in reviewer_reviews.head(3).iterrows():
                    sentiment = review["sentiment_score"]
                    sentiment_emoji = "🟢" if sentiment > 0.05 else ("🔴" if sentiment < -0.05 else "🟡")
                    review_text = str(review["review_text"])[:100] + "..." if len(str(review["review_text"])) > 100 else str(review["review_text"])
                    samples.append(f"{sentiment_emoji} {review_text}")
                
                if samples:
                    sample_reviews_text = "<br>".join(samples)
                else:
                    sample_reviews_text = "No review samples available"
            else:
                sample_reviews_text = "No review samples available"
        else:
            # Fall back to sample_reviews column if it exists
            if "sample_reviews" in reviewer.index and reviewer["sample_reviews"]:
                samples = []
                for review_text in reviewer["sample_reviews"][:3]:
                    if review_text and isinstance(review_text, str):
                        review_preview = review_text[:100] + "..." if len(review_text) > 100 else review_text
                        samples.append(f"💬 {review_preview}")
                sample_reviews_text = "<br>".join(samples) if samples else "No sample reviews"
            else:
                sample_reviews_text = "No sample reviews available"
        
        # Create hover text with reviewer info and sample reviews
        hover_text = (
            f"<b>{reviewer_name}</b><br>"
            f"📊 Reviews: {reviewer['total_reviews']}<br>"
            f"📈 Avg sentiment: {reviewer['avg_sentiment']:+.2f}<br>"
            f"📝 Avg length: {reviewer['avg_review_length_words']:.0f} words<br>"
            f"<br><b>Sample reviews:</b><br>{sample_reviews_text}"
        )
        hover_texts.append(hover_text)
        
        # Store additional data for customdata if needed
        custom_data.append([
            reviewer["avg_sentiment"],
            reviewer["avg_review_length_words"],
            reviewer["total_reviews"]
        ])
    
    # Color by cluster if available
    if "reviewer_cluster" in profiles_df.columns:
        for cluster_id in sorted(profiles_df["reviewer_cluster"].unique()):
            cluster_df = profiles_df[profiles_df["reviewer_cluster"] == cluster_id]
            cluster_name = name_reviewer_cluster(cluster_df)
            
            # Get indices for this cluster to subset hover_texts
            cluster_indices = cluster_df.index
            cluster_hover_texts = [hover_texts[i] for i in cluster_indices]
            cluster_custom_data = [custom_data[i] for i in cluster_indices]
            
            fig.add_trace(go.Scatter(
                x=cluster_df["reviewer_pca_x"],
                y=cluster_df["reviewer_pca_y"],
                mode="markers",
                name=f"Type {cluster_id+1}: {cluster_name}",
                marker=dict(
                    size=cluster_df["total_reviews"] * 3 + 10,
                    color=COLORS[cluster_id % len(COLORS)],
                    opacity=0.7,
                    line=dict(width=1, color="white"),
                ),
                text=cluster_df["reviewer_name"],
                hovertemplate="%{customdata[3]}<extra></extra>",  # Use the pre-formatted hover text
                customdata=list(zip(
                    cluster_df["avg_sentiment"],
                    cluster_df["avg_review_length_words"],
                    cluster_df["total_reviews"],
                    cluster_hover_texts  # Include the formatted hover text
                )),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=profiles_df["reviewer_pca_x"],
            y=profiles_df["reviewer_pca_y"],
            mode="markers",
            marker=dict(
                size=profiles_df["total_reviews"] * 3 + 10,
                color=profiles_df["avg_sentiment"],
                colorscale="RdBu",
                colorbar=dict(title="Avg Sentiment"),
                showscale=True,
                opacity=0.7,
                line=dict(width=1, color="white"),
            ),
            text=profiles_df["reviewer_name"],
            hovertemplate="%{customdata[3]}<extra></extra>",
            customdata=list(zip(
                profiles_df["avg_sentiment"],
                profiles_df["avg_review_length_words"],
                profiles_df["total_reviews"],
                hover_texts
            )),
        ))
    
    # Get variance from separate columns
    variance_pc1 = profiles_df["pca_variance_pc1"].iloc[0] if "pca_variance_pc1" in profiles_df.columns and len(profiles_df) > 0 else 0
    variance_pc2 = profiles_df["pca_variance_pc2"].iloc[0] if "pca_variance_pc2" in profiles_df.columns and len(profiles_df) > 0 else 0
    
    fig.update_layout(
        height=600,
        xaxis=dict(
            title=f"PC1 ({variance_pc1:.0%} variance)" if variance_pc1 else "PC1",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=f"PC2 ({variance_pc2:.0%} variance)" if variance_pc2 else "PC2",
            showgrid=False,
            zeroline=False,
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="monospace",
            bordercolor="#ddd",
            namelength=-1  # Allow long text to wrap
        ),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=20, b=40, l=20, r=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add instructions
    st.caption("💡 **Tip:** Hover over any dot to see the reviewer's profile and sample reviews. Click and drag to zoom, double-click to reset.")


def render_reviewer_clusters(profiles_df: pd.DataFrame):
    """Show detailed breakdown of each reviewer type."""
    st.subheader("Reviewer Personality Types")
    st.caption("Groups of reviewers with similar behavior patterns")
    
    if "reviewer_cluster" not in profiles_df.columns:
        st.info("Cluster reviewers first to see this view.")
        return
    
    # Cluster summaries
    cluster_summaries = []
    for cluster_id in sorted(profiles_df["reviewer_cluster"].unique()):
        cluster_df = profiles_df[profiles_df["reviewer_cluster"] == cluster_id]
        
        # Generate cluster name
        cluster_name = name_reviewer_cluster(cluster_df)
        
        # Calculate topic preferences for this cluster
        topic_cols = [f"topic_{t}" for t in THEME_KEYWORDS.keys()]
        topic_means = {}
        for t in topic_cols:
            if t in cluster_df.columns:
                topic_name = t.replace("topic_", "")
                topic_means[topic_name] = cluster_df[t].mean()
        
        top_topics = sorted(topic_means.items(), key=lambda x: x[1], reverse=True)[:3]
        
        cluster_summaries.append({
            "cluster_id": cluster_id,
            "name": cluster_name,
            "size": len(cluster_df),
            "pct_of_total": len(cluster_df) / len(profiles_df) * 100,
            "avg_sentiment": cluster_df["avg_sentiment"].mean(),
            "avg_length": cluster_df["avg_review_length_words"].mean(),
            "top_topics": ", ".join([f"{t}" for t, _ in top_topics]),
            "sample_reviewers": cluster_df.nlargest(3, "total_reviews")["reviewer_name"].tolist() if len(cluster_df) > 0 else [],
        })
    
    # Display clusters in columns
    cols = st.columns(min(3, len(cluster_summaries)))
    for i, summary in enumerate(cluster_summaries):
        with cols[i % len(cols)]:
            with st.container(border=True):
                st.markdown(f"### {summary['name']}")
                st.markdown(f"**{summary['size']} reviewers** ({summary['pct_of_total']:.0f}%)")
                st.markdown(f"📊 Avg sentiment: {summary['avg_sentiment']:+.2f}")
                st.markdown(f"📝 Avg length: {summary['avg_length']:.0f} words")
                st.markdown(f"🎯 Focus: {summary['top_topics']}")
                st.markdown("**Active reviewers:**")
                for r in summary["sample_reviewers"][:3]:
                    st.markdown(f"- {r}")


def render_top_reviewers(profiles_df: pd.DataFrame):
    """Show leaderboards of most active/influential reviewers."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Most Active Reviewers")
        if len(profiles_df) > 0:
            top_active = profiles_df.nlargest(5, "total_reviews")[
                ["reviewer_name", "total_reviews", "avg_sentiment", "avg_review_length_words"]
            ]
            top_active_display = top_active.copy()
            top_active_display.columns = ["Reviewer", "Reviews", "Avg Sentiment", "Avg Length"]
            top_active_display["Avg Sentiment"] = top_active_display["Avg Sentiment"].apply(lambda x: f"{x:+.2f}")
            top_active_display["Avg Length"] = top_active_display["Avg Length"].apply(lambda x: f"{x:.0f} words")
            st.dataframe(top_active_display, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Most Positive Reviewers")
        if len(profiles_df) > 0:
            cols_to_show = ["reviewer_name", "avg_sentiment", "total_reviews"]
            col_names = ["Reviewer", "Avg Sentiment", "Reviews"]
            
            if "pct_positive" in profiles_df.columns:
                cols_to_show.append("pct_positive")
                col_names.append("% Positive")
            
            top_positive = profiles_df.nlargest(5, "avg_sentiment")[cols_to_show]
            top_positive_display = top_positive.copy()
            top_positive_display.columns = col_names
            top_positive_display["Avg Sentiment"] = top_positive_display["Avg Sentiment"].apply(lambda x: f"{x:+.2f}")
            if "% Positive" in top_positive_display.columns:
                top_positive_display["% Positive"] = top_positive_display["% Positive"].apply(lambda x: f"{x:.0f}%")
            st.dataframe(top_positive_display, use_container_width=True, hide_index=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Most Critical Reviewers")
        if len(profiles_df) > 0:
            cols_to_show = ["reviewer_name", "avg_sentiment", "total_reviews"]
            col_names = ["Reviewer", "Avg Sentiment", "Reviews"]
            
            if "pct_negative" in profiles_df.columns:
                cols_to_show.append("pct_negative")
                col_names.append("% Negative")
            
            top_critical = profiles_df.nsmallest(5, "avg_sentiment")[cols_to_show]
            top_critical_display = top_critical.copy()
            top_critical_display.columns = col_names
            top_critical_display["Avg Sentiment"] = top_critical_display["Avg Sentiment"].apply(lambda x: f"{x:+.2f}")
            if "% Negative" in top_critical_display.columns:
                top_critical_display["% Negative"] = top_critical_display["% Negative"].apply(lambda x: f"{x:.0f}%")
            st.dataframe(top_critical_display, use_container_width=True, hide_index=True)
    
    with col4:
        st.subheader("Most Detailed Reviewers")
        if len(profiles_df) > 0:
            top_detailed = profiles_df.nlargest(5, "avg_review_length_words")[
                ["reviewer_name", "avg_review_length_words", "total_reviews", "avg_sentiment"]
            ]
            top_detailed_display = top_detailed.copy()
            top_detailed_display.columns = ["Reviewer", "Avg Length", "Reviews", "Avg Sentiment"]
            top_detailed_display["Avg Length"] = top_detailed_display["Avg Length"].apply(lambda x: f"{x:.0f} words")
            top_detailed_display["Avg Sentiment"] = top_detailed_display["Avg Sentiment"].apply(lambda x: f"{x:+.2f}")
            st.dataframe(top_detailed_display, use_container_width=True, hide_index=True)


def render_profile_details(profiles_df: pd.DataFrame, original_df: pd.DataFrame):
    """Allow drilling down into individual reviewer profiles."""
    st.subheader("Individual Reviewer Profiles")
    
    # Select reviewer
    reviewer_names = ["All reviewers"] + profiles_df["reviewer_name"].tolist()
    selected = st.selectbox("Select a reviewer to analyze:", reviewer_names)
    
    if selected == "All reviewers":
        # Show summary table
        display_df = profiles_df.copy()
        for col in ["sample_reviews", "reviewer_pca_x", "reviewer_pca_y", "pca_variance"]:
            if col in display_df.columns:
                display_df = display_df.drop(columns=[col])
        st.dataframe(display_df, use_container_width=True)
    else:
        # Show individual profile
        reviewer = profiles_df[profiles_df["reviewer_name"] == selected].iloc[0]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"### {reviewer['reviewer_name']}")
            st.metric("Total Reviews", reviewer["total_reviews"])
            st.metric("Avg Sentiment", f"{reviewer['avg_sentiment']:+.3f}")
            if "avg_rating" in reviewer.index and pd.notna(reviewer["avg_rating"]):
                st.metric("Avg Rating", f"{reviewer['avg_rating']} ★")
            st.metric("Review Length", f"{reviewer['avg_review_length_words']:.0f} words")
        
        with col2:
            # Topic preference radar chart
            topic_cols = [f"topic_{t}" for t in THEME_KEYWORDS.keys()]
            topic_scores = {}
            for t in topic_cols:
                if t in reviewer.index:
                    topic_name = t.replace("topic_", "")
                    topic_scores[topic_name] = reviewer[t]
            
            if topic_scores:
                fig = go.Figure(data=go.Scatterpolar(
                    r=list(topic_scores.values()),
                    theta=list(topic_scores.keys()),
                    fill='toself',
                    marker=dict(color="#2980b9")
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    height=300,
                    margin=dict(t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Show their reviews
        st.markdown("### Their Reviews")
        
        # Try to find their reviews in the original dataset
        if "reviewer_name" in original_df.columns:
            reviewer_reviews = original_df[original_df["reviewer_name"] == selected]
            
            if not reviewer_reviews.empty:
                for _, row in reviewer_reviews.iterrows():
                    sentiment_class = "🟢" if row["sentiment_score"] > 0.05 else ("🔴" if row["sentiment_score"] < -0.05 else "🟡")
                    rating_str = f" · {row['rating']} ★" if "rating" in row and pd.notna(row["rating"]) else ""
                    review_text = str(row["review_text"])[:150] + "..." if len(str(row["review_text"])) > 150 else str(row["review_text"])
                    st.markdown(
                        f"{sentiment_class} **{review_text}**  \n"
                        f"<span style='color: #666; font-size:0.9em'>Sentiment: {row['sentiment_score']:+.2f}{rating_str}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")
            else:
                st.info("No individual reviews available for this reviewer.")
        else:
            # Show sample reviews from profile
            st.markdown("**Sample reviews:**")
            sample_reviews = reviewer.get("sample_reviews", [])
            if sample_reviews and isinstance(sample_reviews, list):
                for r in sample_reviews[:3]:
                    if r and isinstance(r, str):
                        st.markdown(f"💬 {r[:200]}...")
            else:
                st.info("No sample reviews available.")