import pytest
import sys
from unittest.mock import patch, MagicMock

# Mock the whisper module
class MockWhisperModel:
    def transcribe(self, path, **kwargs):
        if path == "/nonexistent/path.wav":
            raise FileNotFoundError("File not found")
        return {"segments": []}

class MockWhisper:
    @staticmethod
    def load_model(model_name):
        return MockWhisperModel()

# Set up mock modules
sys.modules['whisper'] = MockWhisper()

# Now import the module under test
from processing import transcriber

def test_extract_segments_with_missing_keys():
    segments = transcriber.extract_segments({})
    assert segments == []

def test_extract_segments_with_garbage():
    garbage = {"segments": [{"nope": 1, "x": 2}]}
    output = transcriber.extract_segments(garbage)
    # Our implementation now returns segments with empty text rather than skipping them
    assert len(output) == 1
    assert output[0]["text"] == ""  # Text should be empty string
    assert output[0]["start"] == 0  # Default start time
    assert output[0]["end"] == 0    # Default end time

def test_transcribe_wav_raises_on_invalid_path():
    with pytest.raises(Exception):
        transcriber.transcribe_wav("/nonexistent/path.wav")