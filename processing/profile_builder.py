from dataclasses import dataclass, asdict
from collections import defaultdict
from utils.logger import get_logger
from utils.config import CFG
from processing.model_providers import ModelProvider
import json
import re

LOG = get_logger("ProfileBuilder")

@dataclass
class Utterance:
    start: float
    end: float
    speaker: str
    text: str

@dataclass
class Profile:
    speaker: str
    keywords: list[str]
    sentiment: str
    summary: str

def build_profiles(utterances: list[Utterance]) -> list[Profile]:
    """
    Build speaker profiles from utterances
    
    Args:
        utterances: List of utterances with speaker attribution
        
    Returns:
        list[Profile]: Speaker profiles with sentiment, keywords, and summary
    """
    by_speaker = defaultdict(list)
    for u in utterances:
        by_speaker[u.speaker].append(u.text)

    # Get the configured model provider
    provider_name = CFG.get("MODEL_PROVIDER", "openai")
    provider = ModelProvider.get_provider(provider_name)
    LOG.info(f"Building profiles using {provider_name} provider")

    profiles = []
    for speaker, texts in by_speaker.items():
        combined = " ".join(texts)
        
        # Use the model provider for analysis
        sentiment_result = provider.analyze_text(combined, "sentiment")
        keywords_result = provider.analyze_text(combined, "keywords")
        summary_result = provider.analyze_text(combined, "summary")
        
        profile = Profile(
            speaker=speaker,
            keywords=keywords_result.get("keywords", []),
            sentiment=sentiment_result.get("sentiment", "neutral"),
            summary=summary_result.get("summary", combined[:160] + ("..." if len(combined) > 160 else ""))
        )
        profiles.append(profile)

    LOG.info("Built %d profiles", len(profiles))
    return profiles

def profiles_to_json(profiles: list[Profile]) -> list[dict]:
    """Convert a list of Profile objects to JSON-serializable dictionaries"""
    return [asdict(p) for p in profiles]

def asdict(obj):
    """Convert a dataclass object to a dictionary for JSON serialization"""
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__))