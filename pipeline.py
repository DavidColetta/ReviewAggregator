import pandas as pd
import numpy as np
from transformers import pipeline as hf_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from collections import Counter
import re

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

THEME_KEYWORDS = {
    "Food Quality":   ["food", "taste", "delicious", "flavor", "fresh", "bland", "cold", "hot", "cooked", "meal", "dish", "portion", "quality"],
    "Service":        ["service", "staff", "waiter", "waitress", "server", "friendly", "rude", "attentive", "slow", "fast", "helpful", "ignored"],
    "Atmosphere":     ["atmosphere", "ambiance", "vibe", "decor", "noise", "loud", "quiet", "cozy", "clean", "dirty", "comfortable", "music"],
    "Value":          ["price", "expensive", "cheap", "value", "worth", "overpriced", "affordable", "cost", "bill", "money", "pricey"],
    "Wait Time":      ["wait", "slow", "quick", "fast", "long", "minutes", "hour", "reservation", "line", "seated", "delay"],
    "Drinks":         ["drink", "cocktail", "wine", "beer", "coffee", "beverage", "bar", "alcohol", "juice", "tea"],
}

# ─────────────────────────────────────────────
# Load DistilBERT sentiment model (cached)
# ─────────────────────────────────────────────

_sentiment_pipeline = None

def get_sentiment_pipeline():
    """Load once and reuse — first call downloads ~250MB model."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
    return _sentiment_pipeline


# ─────────────────────────────────────────────
# Step 1: Load & validate data
# ─────────────────────────────────────────────

def load_data(source) -> pd.DataFrame:
    """
    Load from a filepath string or a pandas DataFrame.
    Required columns: review_text
    Optional columns: rating, date, reviewer_name
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)

    df = df.dropna(subset=["review_text"])
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].str.len() > 10]  # drop near-empty reviews

    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# Step 2: Extract per-review signals
# ─────────────────────────────────────────────

def get_sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"

def get_theme_scores(text: str) -> dict:
    """Score each theme by keyword hit rate."""
    text_lower = str(text).lower()
    words = re.findall(r'\b\w+\b', text_lower)
    total = len(words) or 1
    return {
        theme: sum(1 for w in words if any(kw in w for kw in kws)) / total
        for theme, kws in THEME_KEYWORDS.items()
    }

def run_distilbert_sentiment(texts: list, progress_callback=None) -> list:
    """
    Run DistilBERT sentiment on a list of texts in batches.
    Returns a list of scores in range [-1, +1].
    POSITIVE label → positive score, NEGATIVE → negative score.
    """
    nlp = get_sentiment_pipeline()
    scores = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = nlp(batch)
        for r in results:
            raw = r["score"]  # confidence 0.5–1.0
            if raw < 0.75:
                # Not confident enough — call it neutral
                scores.append(0.0)
            else:
                score = (raw - 0.5) * 2  # rescale to 0..1
                if r["label"] == "NEGATIVE":
                    score = -score
                scores.append(round(score, 4))

        if progress_callback:
            progress_callback(min(i + batch_size, len(texts)), len(texts))

    return scores

def extract_signals(df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    df = df.copy()

    texts = df["review_text"].tolist()
    df["sentiment_score"] = run_distilbert_sentiment(texts, progress_callback)
    df["sentiment_label"] = df["sentiment_score"].apply(get_sentiment_label)

    theme_scores = df["review_text"].apply(get_theme_scores).apply(pd.Series)
    df = pd.concat([df, theme_scores], axis=1)
    return df


# ─────────────────────────────────────────────
# Step 3: Vectorize reviews for clustering
# ─────────────────────────────────────────────

def vectorize_reviews(df: pd.DataFrame):
    """
    Use TF-IDF to turn review text into vectors, then blend with
    sentiment and theme scores for richer clustering signal.
    """
    tfidf = TfidfVectorizer(
        max_features=200,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    # Handle edge case where min_df=2 is too strict for small datasets
    try:
        tfidf_matrix = tfidf.fit_transform(df["review_text"]).toarray()
    except ValueError:
        tfidf = TfidfVectorizer(max_features=200, stop_words="english", min_df=1)
        tfidf_matrix = tfidf.fit_transform(df["review_text"]).toarray()

    # Theme score matrix
    theme_cols = list(THEME_KEYWORDS.keys())
    theme_matrix = df[theme_cols].values * 5  # upweight themes vs tfidf

    # Sentiment as a column
    sentiment_col = df[["sentiment_score"]].values * 3

    # Stack everything
    combined = np.hstack([tfidf_matrix, theme_matrix, sentiment_col])
    combined = normalize(combined)  # L2 normalize rows

    return combined, tfidf


# ─────────────────────────────────────────────
# Step 4: Cluster reviews
# ─────────────────────────────────────────────

def get_axis_label(component, feature_names, top_n=2):
    loadings = list(zip(feature_names, component))
    sorted_loadings = sorted(loadings, key=lambda x: x[1])
    negative_end = " / ".join(name for name, _ in sorted_loadings[:top_n])
    positive_end = " / ".join(name for name, _ in sorted_loadings[-top_n:])
    return f"← {negative_end}   |   {positive_end} →"


def cluster_reviews(df: pd.DataFrame, n_clusters: int = 5):
    """
    Cluster using K-Means on a higher-dimensional PCA space (up to 10 components),
    then project separately to 2D for visualization only.
    """
    X, tfidf = vectorize_reviews(df)

    # Step 1: Reduce to up to 10 components for clustering (retains more info than 2)
    n_cluster_components = min(10, X.shape[0] - 1, X.shape[1] - 1)
    pca_cluster = PCA(n_components=n_cluster_components, random_state=42)
    X_cluster = pca_cluster.fit_transform(X)

    # Step 2: K-Means on the richer space
    n_clusters = min(n_clusters, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster)

    # Step 3: Separate 2-component PCA just for visualization
    pca_viz = PCA(n_components=2, random_state=42)
    X_viz = pca_viz.fit_transform(X)

    # Step 4: Build axis labels from interpretable features (themes + sentiment)
    theme_names = list(THEME_KEYWORDS.keys())
    try:
        tfidf_names = list(tfidf.get_feature_names_out())
    except Exception:
        tfidf_names = []
    interpretable_names = theme_names + ["sentiment"]
    interpretable_start = len(tfidf_names)

    comp1 = pca_viz.components_[0][interpretable_start:]
    comp2 = pca_viz.components_[1][interpretable_start:]
    axis_x_label = get_axis_label(comp1, interpretable_names)
    axis_y_label = get_axis_label(comp2, interpretable_names)

    variance_explained = pca_viz.explained_variance_ratio_

    pca_meta = {
        "axis_x_label":     axis_x_label,
        "axis_y_label":     axis_y_label,
        "variance_x":       float(variance_explained[0]),
        "variance_y":       float(variance_explained[1]),
        "variance_2d":      float(np.sum(variance_explained)),
        "variance_cluster": float(np.sum(pca_cluster.explained_variance_ratio_)),
        "n_cluster_components": n_cluster_components,
    }

    df = df.copy()
    df["cluster"] = labels
    df["pca_x"] = X_viz[:, 0]
    df["pca_y"] = X_viz[:, 1]

    return df, pca_viz, kmeans, tfidf, pca_meta


# ─────────────────────────────────────────────
# Step 5: Build cluster summaries
# ─────────────────────────────────────────────

def get_top_words(cluster_df: pd.DataFrame, tfidf, n=8) -> list:
    """Get the most representative words for a cluster using TF-IDF."""
    try:
        sub_tfidf = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        sub_tfidf.fit_transform(cluster_df["review_text"])
        scores = dict(zip(sub_tfidf.get_feature_names_out(),
                          sub_tfidf.idf_))
        # Lower IDF = more common in this cluster = more representative
        sorted_words = sorted(scores.items(), key=lambda x: x[1])
        return [w for w, _ in sorted_words[:n]]
    except Exception:
        return []

def name_cluster(cluster_df: pd.DataFrame, top_words: list) -> tuple:
    """Auto-name a cluster by its dominant theme + sentiment. Returns (name, sentiment_tag)."""
    theme_cols = list(THEME_KEYWORDS.keys())
    theme_means = cluster_df[theme_cols].mean().sort_values(ascending=False)
    top_theme = theme_means.index[0]

    avg_sentiment = cluster_df["sentiment_score"].mean()
    if avg_sentiment >= 0.15:
        sentiment_tag = "Praise"
    elif avg_sentiment <= -0.1:
        sentiment_tag = "Complaints"
    else:
        sentiment_tag = "Mixed"

    return f"{top_theme} — {sentiment_tag}", sentiment_tag


def deduplicate_names(summaries: list) -> list:
    """Append top keywords to any clusters that share the same name."""
    from collections import Counter
    name_counts = Counter(s["name"] for s in summaries)
    duplicates = {name for name, count in name_counts.items() if count > 1}
    seen = Counter()
    for s in summaries:
        if s["name"] in duplicates:
            seen[s["name"]] += 1
            keyword_suffix = ", ".join(s["top_words"][:2]) if s["top_words"] else str(seen[s["name"]])
            s["name"] = f"{s['name']} ({keyword_suffix})"
    return summaries


def build_cluster_summaries(df: pd.DataFrame, tfidf) -> list:
    """Return a list of dicts, one per cluster, with summary stats."""
    summaries = []
    for cluster_id in sorted(df["cluster"].unique()):
        cdf = df[df["cluster"] == cluster_id]

        avg_sentiment = cdf["sentiment_score"].mean()
        sentiment_counts = cdf["sentiment_label"].value_counts().to_dict()
        top_words = get_top_words(cdf, tfidf)
        name, sentiment_tag = name_cluster(cdf, top_words)
        avg_rating = cdf["rating"].mean() if "rating" in cdf.columns else None
        sample_reviews = cdf.nlargest(3, "sentiment_score")["review_text"].tolist() + \
                         cdf.nsmallest(2, "sentiment_score")["review_text"].tolist()

        summaries.append({
            "cluster_id":       cluster_id,
            "name":             name,
            "sentiment_tag":    sentiment_tag,
            "review_count":     len(cdf),
            "avg_sentiment":    round(avg_sentiment, 3),
            "avg_rating":       round(avg_rating, 2) if avg_rating is not None else None,
            "sentiment_counts": sentiment_counts,
            "top_words":        top_words,
            "sample_reviews":   sample_reviews[:5],
        })

    summaries = deduplicate_names(summaries)
    return summaries


# ─────────────────────────────────────────────
# Elbow analysis
# ─────────────────────────────────────────────

def compute_elbow_data(source, k_min: int = 2, k_max: int = 8) -> dict:
    """Try K values k_min..k_max, return inertia, silhouette scores, suggested K."""
    from sklearn.metrics import silhouette_score

    df = load_data(source)
    # Only run extract_signals if theme columns aren't already present
    theme_cols = list(THEME_KEYWORDS.keys())
    if not all(c in df.columns for c in theme_cols):
        df = extract_signals(df)
    X, _ = vectorize_reviews(df)

    n_components = min(10, X.shape[0] - 1, X.shape[1] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)

    k_max = min(k_max, len(df) - 1)
    k_values, inertias, silhouettes = [], [], []

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_reduced)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_reduced, labels) if len(set(labels)) > 1 else 0.0
        silhouettes.append(round(sil, 4))
        k_values.append(k)

    silhouette_k = k_values[int(np.argmax(silhouettes))]
    if len(inertias) >= 3:
        deltas = np.diff(inertias)
        second_deriv = np.diff(deltas)
        elbow_k = k_values[1:-1][int(np.argmax(second_deriv))]
    else:
        elbow_k = silhouette_k

    return {
        "k_values":     k_values,
        "inertias":     inertias,
        "silhouettes":  silhouettes,
        "suggested_k":  elbow_k,   # elbow method is the primary recommendation
        "elbow_k":      elbow_k,
        "silhouette_k": silhouette_k,
    }


# ─────────────────────────────────────────────
# Data quality report
# ─────────────────────────────────────────────

def compute_data_quality(raw_df: pd.DataFrame) -> dict:
    """Analyse raw dataframe before cleaning and return a quality report."""
    total_raw = len(raw_df)
    null_text  = raw_df["review_text"].isna().sum() if "review_text" in raw_df.columns else 0
    empty_text = int((raw_df["review_text"].astype(str).str.strip().str.len() <= 10).sum())                  if "review_text" in raw_df.columns else 0
    total_dropped = int(null_text) + empty_text

    texts = raw_df["review_text"].dropna().astype(str)
    lengths = texts.str.len()
    avg_length    = int(lengths.mean())    if len(lengths) else 0
    median_length = int(lengths.median()) if len(lengths) else 0
    short_reviews = int((lengths < 50).sum())
    long_reviews  = int((lengths > 500).sum())

    has_ratings = "rating" in raw_df.columns
    rating_stats = {}
    if has_ratings:
        ratings = pd.to_numeric(raw_df["rating"], errors="coerce").dropna()
        rating_stats = {
            "avg_rating":      round(float(ratings.mean()), 2),
            "pct_5_star":      round((ratings == 5).mean() * 100, 1),
            "pct_1_star":      round((ratings == 1).mean() * 100, 1),
            "missing_ratings": int(raw_df["rating"].isna().sum()),
        }

    date_range = None
    if "date" in raw_df.columns:
        dates = pd.to_datetime(raw_df["date"], errors="coerce").dropna()
        if len(dates):
            date_range = {"earliest": str(dates.min().date()), "latest": str(dates.max().date())}

    return {
        "total_raw":     total_raw,
        "total_clean":   total_raw - total_dropped,
        "total_dropped": total_dropped,
        "null_text":     int(null_text),
        "empty_text":    empty_text,
        "avg_length":    avg_length,
        "median_length": median_length,
        "short_reviews": short_reviews,
        "long_reviews":  long_reviews,
        "has_ratings":   has_ratings,
        "rating_stats":  rating_stats,
        "date_range":    date_range,
    }


# ─────────────────────────────────────────────
# Export helper
# ─────────────────────────────────────────────

def export_results_csv(df: pd.DataFrame, summaries: list) -> str:
    """Return CSV string of review-level data with cluster labels and scores attached."""
    id_to_name = {s["cluster_id"]: s["name"] for s in summaries}
    export_df = df.copy()
    export_df["cluster_name"] = export_df["cluster"].map(id_to_name)

    priority_cols = ["review_text", "rating", "cluster_name", "sentiment_score",
                     "sentiment_label"] + list(THEME_KEYWORDS.keys())
    other_cols = [c for c in export_df.columns
                  if c not in priority_cols and c not in ("cluster", "pca_x", "pca_y")]
    ordered = [c for c in priority_cols if c in export_df.columns] + other_cols
    return export_df[ordered].to_csv(index=False)


# ─────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────

def run_pipeline(source, n_clusters: int = 5, progress_callback=None):
    """
    End-to-end pipeline. Returns (df_with_signals, cluster_summaries).
    source: filepath string or pd.DataFrame
    progress_callback: optional fn(current, total) for progress bars
    """
    df = load_data(source)
    df = extract_signals(df, progress_callback=progress_callback)
    df, pca_viz, kmeans, tfidf, pca_meta = cluster_reviews(df, n_clusters=n_clusters)
    summaries = build_cluster_summaries(df, tfidf)
    return df, summaries, pca_meta


# ─────────────────────────────────────────────
# Sample data generator for testing
# ─────────────────────────────────────────────

def make_sample_data() -> pd.DataFrame:
    import random
    random.seed(99)
    reviews = [
        # Food quality
        ("The pasta was absolutely incredible, fresh and full of flavor. Best I've had in years!", 5),
        ("Food was cold when it arrived and completely bland. Really disappointing meal.", 2),
        ("Delicious dishes, the salmon was cooked perfectly. Generous portions too.", 5),
        ("The burger was dry and overcooked. The fries were soggy. Won't order again.", 1),
        ("Fresh ingredients, amazing taste. The chef clearly knows what they're doing.", 5),
        ("Meal was mediocre at best, nothing special about the flavor at all.", 3),
        ("Best pizza I've ever tasted, the dough was perfect and toppings were fresh.", 5),
        ("Steak was way overcooked despite asking for medium rare. Very disappointing.", 2),
        # Service
        ("Our waiter was incredibly attentive and friendly, made the whole evening special.", 5),
        ("Staff completely ignored us for 20 minutes. Had to ask multiple times for water.", 1),
        ("Service was fast and efficient. Server had great recommendations too.", 4),
        ("Rude staff, felt unwelcome the entire time. Will not be coming back.", 1),
        ("The waitress was so sweet and checked in on us regularly without being intrusive.", 5),
        ("Server seemed distracted and got our order wrong twice.", 2),
        # Wait time
        ("Waited over an hour for our food even though the restaurant was half empty.", 1),
        ("Quick service, food came out in under 15 minutes which was impressive.", 4),
        ("Had a reservation but still waited 40 minutes to be seated. Unacceptable.", 1),
        ("Fast and efficient, no waiting around. In and out in under an hour.", 4),
        ("The wait was completely unreasonable. 90 minute wait for a simple order.", 1),
        # Atmosphere
        ("Beautiful decor, romantic atmosphere, perfect for a date night.", 5),
        ("Way too loud, couldn't hold a conversation. Music was blasting.", 2),
        ("Cozy and comfortable environment. The lighting was perfect.", 5),
        ("Place was dirty, tables were sticky. Doesn't seem like it gets cleaned often.", 1),
        ("Lovely ambiance, very relaxing vibe. We stayed for hours.", 5),
        # Value
        ("Extremely overpriced for what you get. Tiny portions for a lot of money.", 1),
        ("Great value, generous portions and reasonable prices. Will definitely return.", 5),
        ("Prices are way too high, especially compared to the quality of food.", 2),
        ("Very affordable and the food was excellent. Amazing value for money.", 5),
        ("Bill was shocking for what was honestly a mediocre experience.", 2),
        # Drinks
        ("The cocktails were creative and delicious, bartender really knows their craft.", 5),
        ("Wine selection was disappointing and overpriced. Glasses were too small.", 2),
        ("Amazing coffee, best espresso in the city. Always start my morning here.", 5),
        ("Drinks took forever and the cocktail tasted watered down.", 2),
    ]

    rows = []
    for text, rating in reviews:
        rows.append({"review_text": text, "rating": rating})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df, summaries = run_pipeline(make_sample_data(), n_clusters=5)
    for s in summaries:
        print(f"\n[Cluster {s['cluster_id']}] {s['name']}")
        print(f"  Reviews: {s['review_count']} | Sentiment: {s['avg_sentiment']}")
        print(f"  Top words: {', '.join(s['top_words'])}")