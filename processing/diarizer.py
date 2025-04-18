import torch
from pyannote.audio import Pipeline
from utils.logger import get_logger

LOG = get_logger("Diarizer")

def diarize(wav_path):
    try:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=None)
        diar = pipe(wav_path)
        LOG.info("Diarization complete. Speakers: %s", set(track[2] for track in diar.itertracks(yield_label=True)))
        return diar
    except Exception as e:
        LOG.error("Diarization failed: %s", e)
        raise