from collections import Counter
from utils.logger import get_logger
from utils.config import CFG
from processing.model_providers import ModelProvider
import json

LOG = get_logger("Recommender")

def recommend(profiles) -> dict:
    """
    Generate recommendations based on speaker profiles
    
    Args:
        profiles: List of Profile objects or dictionaries
        
    Returns:
        dict: Recommendations based on profile analysis
    """
    # Extract key data from profiles
    keyword_counts = Counter()
    sentiment_counts = Counter()
    all_texts = []

    for p in profiles:
        # Handle both Profile objects and dictionaries
        if hasattr(p, 'keywords'):
            # It's a Profile object
            keyword_counts.update(p.keywords)
            sentiment_counts[p.sentiment] += 1
            all_texts.append(getattr(p, 'summary', ''))
        else:
            # It's a dictionary
            keyword_counts.update(p.get("keywords", []))
            sentiment_counts[p.get("sentiment", "neutral")] += 1
            all_texts.append(p.get("summary", ""))

    dominant_keywords = [kw for kw, _ in keyword_counts.most_common(5)]
    dominant_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else "neutral"

    # Determine if we should use AI-enhanced recommendations
    use_ai_enhanced = CFG.get("USE_AI_ENHANCED_RECOMMENDATIONS", False)
    
    if use_ai_enhanced:
        try:
            # Get the configured model provider
            provider_name = CFG.get("MODEL_PROVIDER", "openai")
            provider = ModelProvider.get_provider(provider_name)
            LOG.info(f"Generating AI-enhanced recommendations using {provider_name} provider")
            
            # Combine all text for comprehensive analysis
            combined_text = "\n".join(all_texts)
            
            # Use the model to generate recommendations
            prompt = f"""
            Based on the following focus group discussion summaries, provide marketing recommendations.
            The dominant sentiment is: {dominant_sentiment}
            The key topics are: {', '.join(dominant_keywords)}

            Discussion summaries:
            {combined_text}
            
            Provide recommendations for tone, keywords, color scheme, and messaging focus.
            """
            
            result = provider.analyze_text(prompt, "custom")
            
            # Check if we got a valid response
            if isinstance(result, dict) and not result.get("error"):
                # Extract structured recommendations if available
                ai_suggestions = result.get("result", {})
                if isinstance(ai_suggestions, dict):
                    LOG.info("Using AI-enhanced recommendations")
                    return ai_suggestions
            
            LOG.warning("AI recommendations failed, falling back to rule-based")
        except Exception as e:
            LOG.error(f"Error generating AI recommendations: {e}")
            LOG.warning("Falling back to rule-based recommendations")
    
    # Default rule-based recommendations
    suggestions = {
        "tone": _suggest_tone(dominant_sentiment),
        "keywords": dominant_keywords,
        "color_scheme": _suggest_colors(dominant_sentiment),
        "focus": _suggest_focus(dominant_keywords)
    }

    LOG.info("Generated rule-based recommendations: %s", suggestions)
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