from collections import Counter
from utils.logger import get_logger

LOG = get_logger("Recommender")

def recommend(profiles: list[dict]) -> dict:
    keyword_counts = Counter()
    sentiment_counts = Counter()

    for p in profiles:
        keyword_counts.update(p.get("keywords", []))
        sentiment_counts[p.get("sentiment", "neutral")] += 1

    dominant_keywords = [kw for kw, _ in keyword_counts.most_common(5)]
    dominant_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else "neutral"

    suggestions = {
        "tone": _suggest_tone(dominant_sentiment),
        "keywords": dominant_keywords,
        "color_scheme": _suggest_colors(dominant_sentiment),
        "focus": _suggest_focus(dominant_keywords)
    }

    LOG.info("Generated recommendations: %s", suggestions)
    return suggestions

def _suggest_tone(sentiment: str) -> str:
    return {
        "positive": "inspirational and bold",
        "negative": "reassuring and gentle",
        "neutral": "professional and clean"
    }.get(sentiment, "neutral")

def _suggest_colors(sentiment: str) -> str:
    return {
        "positive": "vibrant (orange, green, blue)",
        "negative": "calm (gray, beige, soft blue)",
        "neutral": "monochrome with accents"
    }.get(sentiment, "neutral palette")

def _suggest_focus(keywords: list[str]) -> str:
    return f"Messaging should emphasize: {', '.join(keywords)}"