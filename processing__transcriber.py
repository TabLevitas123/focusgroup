from pathlib import Path
import json
import whisper
from utils.logger import get_logger

LOG = get_logger("Transcriber")

def transcribe_wav(path: Path) -> dict:
    model = whisper.load_model("base")
    result = model.transcribe(str(path), language="en")
    LOG.info("Transcription complete. Segments: %d", len(result.get("segments", [])))
    return result

def extract_segments(transcription: dict) -> list[dict]:
    segments = transcription.get("segments", [])
    return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments]