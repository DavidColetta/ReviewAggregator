# [FIXED FILE] profile_analyzer.py

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import re
import streamlit as st

from pipeline import THEME_KEYWORDS, get_sentiment_label
from views.components import COLORS


# ─────────────────────────────────────────────
# Profile scraping & parsing
# ─────────────────────────────────────────────

def extract_reviewer_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract reviewer information from a dataframe of reviews.
    
    IMPORTANT: Without real reviewer identifiers, we treat each review as
    coming from a unique reviewer. This avoids falsely grouping different
    people together.
    
    If reviewer_name column exists (from Trustpilot), use it to identify
    real unique reviewers.
    """
    df = df.copy()
    
    # Check if we have real reviewer names (from Trustpilot)
    if "reviewer_name" in df.columns:
        # Use real reviewer names as identifiers
        df["reviewer_id"] = df["reviewer_name"]
        
        # Count how many real repeat reviewers we have
        unique_reviewers = df["reviewer_name"].nunique()
        total_reviews = len(df)
        repeat_rate = (total_reviews - unique_reviewers) / total_reviews * 100
        
        print(f"📊 Found {unique_reviewers} unique reviewers across {total_reviews} reviews")
        print(f"📈 Repeat reviewer rate: {repeat_rate:.1f}%")
        
    else:
        # No real reviewer data - treat each review as unique
        # This is honest and avoids false groupings
        df["reviewer_id"] = range(len(df))
        df["reviewer_name"] = [f"Review_{i}" for i in range(len(df))]
        
        print("⚠️ No reviewer names found - treating each review as a unique reviewer")
        print("   Profile analysis will show per-review patterns, not per-person patterns")
    
    return df


def analyze_reviewers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to analyze all reviewers in a dataset.
    Returns a dataframe with one row per reviewer and their profile metrics.
    
    Note: If no real reviewer identifiers exist, each review is treated as
    its own "reviewer" to avoid false groupings.
    """
    # Ensure we have reviewer IDs (honestly assigned)
    df = extract_reviewer_profiles(df)
    
    # Check if we're using real reviewer data or synthetic unique IDs
    using_real_data = "reviewer_name" in df.columns and df["reviewer_name"].nunique() < len(df)
    
    reviewer_profiles = []
    
    for reviewer_id, group in df.groupby("reviewer_id"):
        reviewer_name = group["reviewer_name"].iloc[0]
        
        # Basic stats
        profile = {
            "reviewer_id": reviewer_id,
            "reviewer_name": reviewer_name,
            "total_reviews": len(group),
            "is_repeat_reviewer": len(group) > 1,  # Flag for real repeat reviewers
            "first_review_date": group["date"].min() if "date" in group.columns else None,
            "last_review_date": group["date"].max() if "date" in group.columns else None,
        }
        
        # Add style metrics
        style_metrics = get_reviewer_style_metrics(group["review_text"])
        profile.update(style_metrics)
        
        # Add topic preferences
        topic_prefs = get_reviewer_topic_preferences(group["review_text"])
        for topic, score in topic_prefs.items():
            profile[f"topic_{topic}"] = score
        
        # Add sentiment profile
        sentiment_profile = get_reviewer_sentiment_profile(group)
        profile.update(sentiment_profile)
        
        # Store sample reviews
        sample_reviews = group.nlargest(3, "sentiment_score")["review_text"].tolist() + \
                         group.nsmallest(2, "sentiment_score")["review_text"].tolist()
        profile["sample_reviews"] = sample_reviews[:3]
        
        reviewer_profiles.append(profile)
    
    result_df = pd.DataFrame(reviewer_profiles)
    
    # Add a note about data quality
    if not using_real_data:
        result_df.attrs["data_quality"] = "synthetic"
        result_df.attrs["warning"] = "No real reviewer identifiers - each review treated as unique"
    else:
        result_df.attrs["data_quality"] = "real"
    
    return result_df


def get_reviewer_style_metrics(reviews_series: pd.Series) -> dict:
    """Calculate style metrics for a reviewer's reviews."""
    texts = reviews_series.tolist()
    
    # Length metrics
    lengths = [len(str(t).split()) for t in texts]
    avg_length = np.mean(lengths) if lengths else 0
    
    # Punctuation usage (excitement/question markers)
    exclamation_count = sum(str(t).count('!') for t in texts)
    question_count = sum(str(t).count('?') for t in texts)
    
    # Caps usage (shouting)
    caps_ratio = sum(
        sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1)
        for t in texts
    ) / max(len(texts), 1)
    
    return {
        "avg_review_length_words": round(avg_length, 1),
        "exclamation_rate": round(exclamation_count / max(len(texts), 1), 2),
        "question_rate": round(question_count / max(len(texts), 1), 2),
        "caps_ratio": round(caps_ratio, 3),
        "review_count": len(texts),
    }


def get_reviewer_topic_preferences(reviews_series: pd.Series) -> dict:
    """
    Calculate which topics a reviewer tends to mention.
    Returns normalized topic preference scores.
    """
    topic_mentions = defaultdict(int)
    
    for text in reviews_series:
        text_lower = str(text).lower()
        for topic, keywords in THEME_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                topic_mentions[topic] += 1
    
    # Normalize by review count
    total_reviews = len(reviews_series)
    if total_reviews > 0:
        return {topic: round(count / total_reviews, 3) 
                for topic, count in topic_mentions.items()}
    return {topic: 0 for topic in THEME_KEYWORDS.keys()}


def get_reviewer_sentiment_profile(reviews_df: pd.DataFrame) -> dict:
    """
    Analyze a reviewer's sentiment patterns across their reviews.
    """
    sentiments = reviews_df["sentiment_score"].tolist()
    labels = reviews_df["sentiment_label"].tolist() if "sentiment_label" in reviews_df.columns else []
    
    profile = {
        "avg_sentiment": round(np.mean(sentiments), 3),
        "sentiment_std": round(np.std(sentiments), 3),
        "sentiment_trend": "consistent" if np.std(sentiments) < 0.2 else "variable",
    }
    
    if labels:
        label_counts = Counter(labels)
        profile.update({
            "pct_positive": round(label_counts.get("Positive", 0) / len(labels) * 100, 1),
            "pct_negative": round(label_counts.get("Negative", 0) / len(labels) * 100, 1),
            "pct_neutral": round(label_counts.get("Neutral", 0) / len(labels) * 100, 1),
        })
    
    # Rating pattern if available
    if "rating" in reviews_df.columns and reviews_df["rating"].notna().any():
        ratings = reviews_df["rating"].dropna()
        profile.update({
            "avg_rating": round(ratings.mean(), 2),
            "rating_std": round(ratings.std(), 2),
            "is_strict": ratings.mean() < 3.0 if len(ratings) > 0 else False,  # Tends to give low ratings
            "is_generous": ratings.mean() > 4.0 if len(ratings) > 0 else False,  # Tends to give high ratings
        })
    
    return profile


def analyze_reviewers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to analyze all reviewers in a dataset.
    Returns a dataframe with one row per reviewer and their profile metrics.
    """
    # Ensure we have reviewer IDs
    df = extract_reviewer_profiles(df)
    
    reviewer_profiles = []
    
    for reviewer_id, group in df.groupby("reviewer_id"):
        reviewer_name = group["reviewer_name"].iloc[0]
        
        # Basic stats
        profile = {
            "reviewer_id": reviewer_id,
            "reviewer_name": reviewer_name,
            "total_reviews": len(group),
            "first_review_date": group["date"].min() if "date" in group.columns else None,
            "last_review_date": group["date"].max() if "date" in group.columns else None,
        }
        
        # Add style metrics
        style_metrics = get_reviewer_style_metrics(group["review_text"])
        profile.update(style_metrics)
        
        # Add topic preferences
        topic_prefs = get_reviewer_topic_preferences(group["review_text"])
        for topic, score in topic_prefs.items():
            profile[f"topic_{topic}"] = score
        
        # Add sentiment profile
        sentiment_profile = get_reviewer_sentiment_profile(group)
        profile.update(sentiment_profile)
        
        # Store sample reviews
        sample_reviews = group.nlargest(3, "sentiment_score")["review_text"].tolist() + \
                         group.nsmallest(2, "sentiment_score")["review_text"].tolist()
        profile["sample_reviews"] = sample_reviews[:3]  # Just top 3 for profile
        
        reviewer_profiles.append(profile)
    
    return pd.DataFrame(reviewer_profiles)


def cluster_reviewers(profiles_df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Cluster reviewers based on their behavior patterns.
    Uses topic preferences + sentiment patterns + style metrics.
    """
    # Select features for clustering
    topic_cols = [f"topic_{t}" for t in THEME_KEYWORDS.keys()]
    style_cols = ["avg_review_length_words", "exclamation_rate", "caps_ratio"]
    sentiment_cols = ["avg_sentiment", "sentiment_std"]
    
    available_cols = [c for c in topic_cols + style_cols + sentiment_cols 
                     if c in profiles_df.columns]
    
    if len(available_cols) < 3:
        # Not enough features, return without clustering
        result = profiles_df.copy()
        result["reviewer_cluster"] = 0
        result["reviewer_pca_x"] = 0
        result["reviewer_pca_y"] = 0
        # Store variance as a scalar or None instead of a list
        result["pca_variance_pc1"] = 0.0
        result["pca_variance_pc2"] = 0.0
        return result
    
    X = profiles_df[available_cols].fillna(0).values
    
    # Normalize
    X_scaled = StandardScaler().fit_transform(X)
    
    # Cluster
    n_clusters = min(n_clusters, len(profiles_df) - 1, 6)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Reduce for visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    
    result = profiles_df.copy()
    result["reviewer_cluster"] = labels
    result["reviewer_pca_x"] = coords[:, 0]
    result["reviewer_pca_y"] = coords[:, 1]
    
    # Store variance as separate columns instead of a list
    variance_ratios = pca.explained_variance_ratio_
    result["pca_variance_pc1"] = variance_ratios[0] if len(variance_ratios) > 0 else 0.0
    result["pca_variance_pc2"] = variance_ratios[1] if len(variance_ratios) > 1 else 0.0
    
    return result


def name_reviewer_cluster(cluster_df: pd.DataFrame) -> str:
    """
    Generate a descriptive name for a cluster of reviewers based on their dominant traits.
    """
    if len(cluster_df) == 0:
        return "Empty Cluster"
    
    traits = []
    
    # Check sentiment tendency
    avg_sent = cluster_df["avg_sentiment"].mean()
    if avg_sent > 0.2:
        traits.append("Optimistic")
    elif avg_sent < -0.1:
        traits.append("Critical")
    else:
        traits.append("Balanced")
    
    # Check topic focus
    topic_cols = [f"topic_{t}" for t in THEME_KEYWORDS.keys()]
    topic_means = {}
    for t in topic_cols:
        if t in cluster_df.columns:
            topic_name = t.replace("topic_", "")
            topic_means[topic_name] = cluster_df[t].mean()
    
    if topic_means:
        top_topic = max(topic_means.items(), key=lambda x: x[1])[0]
        traits.append(f"{top_topic}-focused")
    
    # Check writing style
    if "avg_review_length_words" in cluster_df.columns:
        avg_length = cluster_df["avg_review_length_words"].mean()
        if avg_length > 100:
            traits.append("Detailed")
        elif avg_length < 30:
            traits.append("Brief")
    
    # Check consistency
    if "sentiment_std" in cluster_df.columns:
        if cluster_df["sentiment_std"].mean() < 0.15:
            traits.append("Consistent")
    
    return " · ".join(traits[:3])  # Keep it concise