import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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

sentiment_analyzer = SentimentIntensityAnalyzer()


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

def get_sentiment(text: str) -> float:
    return sentiment_analyzer.polarity_scores(str(text))["compound"]

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

def extract_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment_score"] = df["review_text"].apply(get_sentiment)
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

def cluster_reviews(df: pd.DataFrame, n_clusters: int = 5):
    """Run PCA + KMeans on reviews. Returns df with cluster/pca columns added."""
    X, tfidf = vectorize_reviews(df)

    # PCA for visualization
    n_components = min(2, X.shape[0] - 1, X.shape[1] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    # KMeans
    n_clusters = min(n_clusters, len(df))  # can't have more clusters than reviews
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    df = df.copy()
    df["cluster"] = labels
    df["pca_x"] = X_pca[:, 0]
    df["pca_y"] = X_pca[:, 1] if X_pca.shape[1] > 1 else 0.0

    return df, pca, kmeans, tfidf


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

def name_cluster(cluster_df: pd.DataFrame) -> str:
    """Auto-name a cluster by its dominant theme + sentiment."""
    theme_cols = list(THEME_KEYWORDS.keys())
    theme_means = cluster_df[theme_cols].mean()
    top_theme = theme_means.idxmax()

    avg_sentiment = cluster_df["sentiment_score"].mean()
    if avg_sentiment >= 0.2:
        sentiment_tag = "Praise"
    elif avg_sentiment <= -0.1:
        sentiment_tag = "Complaints"
    else:
        sentiment_tag = "Mixed Feedback"

    return f"{top_theme} — {sentiment_tag}"

def build_cluster_summaries(df: pd.DataFrame, tfidf) -> list:
    """Return a list of dicts, one per cluster, with summary stats."""
    summaries = []
    for cluster_id in sorted(df["cluster"].unique()):
        cdf = df[df["cluster"] == cluster_id]

        avg_sentiment = cdf["sentiment_score"].mean()
        sentiment_counts = cdf["sentiment_label"].value_counts().to_dict()
        top_words = get_top_words(cdf, tfidf)
        name = name_cluster(cdf)
        avg_rating = cdf["rating"].mean() if "rating" in cdf.columns else None
        sample_reviews = cdf.nlargest(3, "sentiment_score")["review_text"].tolist() + \
                         cdf.nsmallest(2, "sentiment_score")["review_text"].tolist()

        summaries.append({
            "cluster_id":      cluster_id,
            "name":            name,
            "review_count":    len(cdf),
            "avg_sentiment":   round(avg_sentiment, 3),
            "avg_rating":      round(avg_rating, 2) if avg_rating is not None else None,
            "sentiment_counts": sentiment_counts,
            "top_words":       top_words,
            "sample_reviews":  sample_reviews[:5],
        })

    return summaries


# ─────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────

def run_pipeline(source, n_clusters: int = 5):
    """
    End-to-end pipeline. Returns (df_with_signals, cluster_summaries).
    source: filepath string or pd.DataFrame
    """
    df = load_data(source)
    df = extract_signals(df)
    df, pca, kmeans, tfidf = cluster_reviews(df, n_clusters=n_clusters)
    summaries = build_cluster_summaries(df, tfidf)
    return df, summaries


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