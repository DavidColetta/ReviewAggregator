"""
Compare Businesses view — add companies, PCA landscape, and comparison charts.
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

from pipeline import THEME_KEYWORDS, load_data, extract_signals
from views.components import COLORS
from utils.charts import base_layout, topic_group_bars, sentiment_bar


# ─────────────────────────────────────────────
# Cached scraping
# ─────────────────────────────────────────────

@st.cache_data
def scrape_and_process(url: str, max_pages: int) -> dict:
    """Scrape a Trustpilot URL and return an aggregated company feature dict."""
    from scraper import scrape_trustpilot
    df, name = scrape_trustpilot(url, max_pages=max_pages)
    df = load_data(df)
    df = extract_signals(df)

    theme_cols = list(THEME_KEYWORDS.keys())
    feature = {
        "name":          name,
        "review_count":  len(df),
        "avg_sentiment": round(float(df["sentiment_score"].mean()), 3),
        "avg_rating":    round(float(df["rating"].mean()), 2) if "rating" in df.columns else None,
        "pct_positive":  round(float((df["sentiment_label"] == "Positive").mean()) * 100, 1),
        "pct_negative":  round(float((df["sentiment_label"] == "Negative").mean()) * 100, 1),
        "sample_reviews": (
            df.nlargest(3, "sentiment_score")["review_text"].tolist() +
            df.nsmallest(2, "sentiment_score")["review_text"].tolist()
        ),
    }
    for tc in theme_cols:
        feature[tc] = round(float(df[tc].mean()), 4)
        # Sentiment among reviews that meaningfully mention this topic (score > 0.1)
        mentioning = df[df[tc] > 0]  # any keyword hit counts as a mention
        if len(mentioning) > 0:
            feature[f"{tc}_sentiment"] = round(float(mentioning["sentiment_score"].mean()), 3)
        else:
            feature[f"{tc}_sentiment"] = 0.0

    return feature


# ─────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────

def render(add_clicked: bool, compare_urls: list, compare_pages: int):
    """Entry point called from app.py."""

    st.markdown("# ⚖️ Compare Businesses")
    st.caption("Add companies via Trustpilot URLs. Each company becomes a bubble on the map.")

    # ── Handle add button — process all URLs ──
    if add_clicked and compare_urls:
        if "compare_companies" not in st.session_state:
            st.session_state.compare_companies = []
        existing = [c["name"] for c in st.session_state.compare_companies]

        progress = st.progress(0, text=f"Scraping 0 / {len(compare_urls)} companies...")
        for idx, url in enumerate(compare_urls):
            progress.progress(idx / len(compare_urls), text=f"Scraping {url}...")
            try:
                feature = scrape_and_process(url, compare_pages)
                if feature["name"] in existing:
                    st.warning(f"**{feature['name']}** is already added — skipped.")
                else:
                    st.session_state.compare_companies.append(feature)
                    existing.append(feature["name"])
                    st.success(f"Added **{feature['name']}** ({feature['review_count']} reviews)")
            except Exception as e:
                st.error(f"Failed to scrape {url}: {e}")
        progress.empty()

    companies = st.session_state.get("compare_companies", [])

    if len(companies) < 2:
        remaining = 2 - len(companies)
        noun = "company" if remaining == 1 else "companies"
        st.info(f"👈 Add at least {remaining} more {noun} using the sidebar to start comparing.")
        return

    # ── Companies list — rendered here so it always reflects latest state ──
    st.markdown("**Companies in comparison:**")
    for i, co in enumerate(st.session_state.compare_companies):
        ccol1, ccol2 = st.columns([6, 1])
        ccol1.info(f"📍 **{co['name']}** — {co['review_count']} reviews")
        if ccol2.button("✕ Remove", key=f"remove_{i}"):
            st.session_state.compare_companies.pop(i)
            st.rerun()

    st.markdown("---")

    # ── Build aggregated dataframe ──
    theme_cols = list(THEME_KEYWORDS.keys())
    co_df = pd.DataFrame(companies)
    co_df = _add_pca_coords(co_df, theme_cols)
    variance = co_df.attrs.get("variance", [0.0, 0.0])
    feature_cols = co_df.attrs.get("feature_cols", [])

    # ── Tabs ──
    ctab1, ctab2, ctab3, ctab4 = st.tabs([
        "🗺️ Company Map", "📊 Topic Breakdown",
        "📈 Sentiment", "🏆 Rankings"
    ])

    with ctab1:
        _tab_map(co_df, variance, feature_cols)

    with ctab2:
        _tab_topics(co_df, theme_cols)

    with ctab3:
        _tab_sentiment(co_df)

    with ctab4:
        _tab_rankings(co_df)


# ─────────────────────────────────────────────
# PCA helper
# ─────────────────────────────────────────────

def _add_pca_coords(co_df: pd.DataFrame, theme_cols: list) -> pd.DataFrame:
    # Use per-topic sentiment scores + overall sentiment + rating.
    # This means position reflects *how companies are perceived* on each topic,
    # not just how often they mention it.
    sentiment_cols = [f"{tc}_sentiment" for tc in theme_cols]
    candidate_cols = ["avg_sentiment", "avg_rating"] + sentiment_cols
    feature_cols = [c for c in candidate_cols if c in co_df.columns and co_df[c].notna().all()]

    X = co_df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    n_components = min(2, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)

    co_df = co_df.copy()
    co_df["pca_x"] = coords[:, 0]
    co_df["pca_y"] = coords[:, 1] if coords.shape[1] > 1 else 0.0
    co_df.attrs["variance"] = pca.explained_variance_ratio_.tolist()
    co_df.attrs["feature_cols"] = feature_cols  # expose for display
    return co_df


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

def _tab_map(co_df: pd.DataFrame, variance: list, feature_cols: list = None):
    st.subheader("Company Landscape")
    features_str = ", ".join(feature_cols) if feature_cols else "sentiment & topic scores"
    st.caption(
        f"Position reflects review profile similarity across: **{features_str}**. "
        f"Chart captures {sum(variance):.0%} of variance. Size = review count."
    )

    fig = go.Figure()
    max_count = co_df["review_count"].max()

    for i, row in co_df.iterrows():
        reviews_text = "<br>".join(
            f'• {r[:90]}{"…" if len(r) > 90 else ""}'
            for r in row["sample_reviews"][:3]
        )
        rating_str = f"{row['avg_rating']} ★  ·  " if row.get("avg_rating") else ""
        hover = (
            f"<b>{row['name']}</b><br>"
            f"{row['review_count']} reviews  ·  {rating_str}"
            f"sentiment {row['avg_sentiment']:+.2f}<br><br>"
            f"{reviews_text}"
        )
        fig.add_trace(go.Scatter(
            x=[row["pca_x"]], y=[row["pca_y"]],
            mode="markers+text",
            marker=dict(
                size=row["review_count"],
                sizemode="area",
                sizeref=2.0 * max_count / (80 ** 2),
                color=COLORS[i % len(COLORS)],
                opacity=0.75,
                line=dict(width=2, color="white"),
            ),
            text=row["name"],
            textposition="top center",
            textfont=dict(size=12, color="black", family="Source Sans 3"),
            hovertemplate=hover + "<extra></extra>",
            name=row["name"],
        ))

    x_title = f"PC1 ({variance[0]:.0%} variance)" if variance else ""
    y_title = f"PC2 ({variance[1]:.0%} variance)" if len(variance) > 1 else ""

    fig.update_layout(**base_layout(
        height=560,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title=dict(text=x_title, font=dict(size=11, color="#888"))),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title=dict(text=y_title, font=dict(size=11, color="#888"))),
        hoverlabel=dict(bgcolor="white", font_size=13, bordercolor="#ddd"),
        margin=dict(t=20, b=40, l=20, r=20),
    ))
    st.plotly_chart(fig, use_container_width=True)


def _tab_topics(co_df: pd.DataFrame, theme_cols: list):
    st.subheader("Topic Sentiment & Volume")
    st.caption(
        "Each bubble is one company × topic. "
        "**Y axis** = sentiment (higher = more positive). "
        "**Size** = how often the topic is mentioned. "
        "Hover for exact values."
    )

    import plotly.graph_objects as go
    import numpy as np

    n_companies = len(co_df)
    # Spread companies within each topic column using evenly spaced offsets
    # so bubbles never overlap regardless of count
    offsets = np.linspace(-0.3, 0.3, n_companies) if n_companies > 1 else [0.0]

    # Compute global max mention rate for consistent bubble sizing
    all_mention_rates = [
        row.get(tc, 0.0)
        for _, row in co_df.iterrows()
        for tc in theme_cols
    ]
    max_mention = max(all_mention_rates) or 1.0

    fig = go.Figure()

    # Shaded sentiment regions
    fig.add_hrect(y0=0, y1=1.1, fillcolor="#27ae60", opacity=0.04, line_width=0)
    fig.add_hrect(y0=-1.1, y1=0, fillcolor="#c0392b", opacity=0.04, line_width=0)
    fig.add_hline(y=0, line_dash="dash", line_color="#ccc", line_width=1)

    for i, (_, row) in enumerate(co_df.iterrows()):
        x_vals, y_vals, sizes, hovers = [], [], [], []

        for j, tc in enumerate(theme_cols):
            mention_rate = row.get(tc, 0.0)
            sentiment    = row.get(f"{tc}_sentiment", 0.0)
            x_vals.append(j + offsets[i])       # topic index + company offset
            y_vals.append(sentiment)
            # Scale bubble size: min 10px, max 50px
            size = 10 + 40 * (mention_rate / max_mention)
            sizes.append(size)
            hovers.append(
                f"<b>{row['name']}</b> — {tc}<br>"
                f"Sentiment: {sentiment:+.3f}<br>"
                f"Mention rate: {mention_rate:.3f}"
            )

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            name=row["name"],
            marker=dict(
                size=sizes,
                color=COLORS[i % len(COLORS)],
                opacity=0.8,
                line=dict(width=1.5, color="white"),
            ),
            hovertemplate=[h + "<extra></extra>" for h in hovers],
        ))

    fig.update_layout(**base_layout(
        height=500,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(theme_cols))),
            ticktext=theme_cols,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Sentiment among mentions",
            range=[-1.1, 1.1],
            zeroline=False,
            showgrid=True, gridcolor="#eee",
        ),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=20, b=60, l=20, r=20),
        annotations=[
            dict(xref="paper", yref="paper", x=1.01, y=0.75,
                 text="▲ Praised", showarrow=False,
                 font=dict(color="#27ae60", size=11), xanchor="left"),
            dict(xref="paper", yref="paper", x=1.01, y=0.22,
                 text="▼ Criticised", showarrow=False,
                 font=dict(color="#c0392b", size=11), xanchor="left"),
        ],
    ))

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bigger bubble = topic mentioned more. Hover any bubble for exact figures.")


def _tab_sentiment(co_df: pd.DataFrame):
    st.subheader("Sentiment Comparison")
    scol1, scol2 = st.columns(2)

    with scol1:
        fig = sentiment_bar(
            co_df["name"].tolist(),
            co_df["avg_sentiment"].tolist(),
        )
        fig.update_layout(title="Average Sentiment Score")
        st.plotly_chart(fig, use_container_width=True)

    with scol2:
        pct_colors = [COLORS[i % len(COLORS)] for i in range(len(co_df))]
        fig_pos = go.Figure(go.Bar(
            x=co_df["name"],
            y=co_df["pct_positive"],
            marker_color=pct_colors,
            text=[f"{v:.0f}%" for v in co_df["pct_positive"]],
            textposition="outside",
        ))
        fig_pos.update_layout(**base_layout(
            title="% Positive Reviews",
            height=360,
            yaxis=dict(range=[0, 110]),
            margin=dict(t=40, b=10),
        ))
        st.plotly_chart(fig_pos, use_container_width=True)


def _tab_rankings(co_df: pd.DataFrame):
    st.subheader("Company Rankings")

    rank_df = co_df[["name", "avg_sentiment", "avg_rating", "pct_positive", "review_count"]].copy()
    rank_df.columns = ["Company", "Avg Sentiment", "Avg Rating", "% Positive", "Reviews"]
    rank_df["Avg Rating"]   = rank_df["Avg Rating"].apply(lambda x: f"{x} ★" if x else "—")
    rank_df["% Positive"]   = rank_df["% Positive"].apply(lambda x: f"{x:.0f}%")
    rank_df["Avg Sentiment"] = rank_df["Avg Sentiment"].apply(lambda x: f"{x:+.3f}")
    rank_df = rank_df.sort_values("% Positive", ascending=False).reset_index(drop=True)
    rank_df.index += 1

    st.dataframe(rank_df, use_container_width=True)

    csv = co_df.drop(columns=["sample_reviews", "pca_x", "pca_y"], errors="ignore").to_csv(index=False)
    st.download_button(
        "⬇️ Download comparison CSV",
        data=csv,
        file_name="company_comparison.csv",
        mime="text/csv",
    )