from pathlib import Path
import json
from utils.logger import get_logger
from utils.config import CFG
from processing.model_providers import ModelProvider

LOG = get_logger("Transcriber")

def transcribe_wav(path: Path) -> dict:
    """
    Transcribe audio using the configured model provider
    
    Args:
        path: Path to the audio file
        
    Returns:
        dict: Transcription result with text and segments
    """
    provider_name = CFG.get("MODEL_PROVIDER", "openai")
    LOG.info(f"Transcribing using {provider_name} provider")
    
    provider = ModelProvider.get_provider(provider_name)
    result = provider.transcribe(str(path))
    LOG.info("Transcription complete. Segments: %d", len(result.get("segments", [])))
    return result

def extract_segments(transcription: dict) -> list[dict]:
    segments = transcription.get("segments", [])
    result = []
    for s in segments:
        try:
            # Only add segments that have all required keys
            segment = {
                "start": s.get("start", 0),
                "end": s.get("end", 0),
                "text": s.get("text", "")
            }
            result.append(segment)
        except Exception as e:
            LOG.warning(f"Skipping invalid segment: {e}")
    return result