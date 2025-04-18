from dataclasses import dataclass, asdict
from collections import defaultdict
from utils.logger import get_logger
import json

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
    by_speaker = defaultdict(list)
    for u in utterances:
        by_speaker[u.speaker].append(u.text)

    profiles = []
    for speaker, texts in by_speaker.items():
        combined = " ".join(texts)
        profile = Profile(
            speaker=speaker,
            keywords=_extract_keywords(combined),
            sentiment=_guess_sentiment(combined),
            summary=_summarize(combined)
        )
        profiles.append(profile)

    LOG.info("Built %d profiles", len(profiles))
    return profiles

def _extract_keywords(text: str) -> list[str]:
    return sorted(set(word.lower() for word in text.split() if len(word) > 5))[:5]

def _guess_sentiment(text: str) -> str:
    if "love" in text or "great" in text:
        return "positive"
    if "hate" in text or "bad" in text:
        return "negative"
    return "neutral"

def _summarize(text: str) -> str:
    return text[:160] + ("..." if len(text) > 160 else "")

def profiles_to_json(profiles: list[Profile]) -> list[dict]:
    return [asdict(p) for p in profiles]

def asdict(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__))