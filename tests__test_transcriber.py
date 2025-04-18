from processing import transcriber
import pytest

def test_extract_segments_with_missing_keys():
    segments = transcriber.extract_segments({})
    assert segments == []

def test_extract_segments_with_garbage():
    garbage = {"segments": [{"nope": 1, "x": 2}]}
    output = transcriber.extract_segments(garbage)
    assert all("text" in s for s in output) is False

def test_transcribe_wav_raises_on_invalid_path():
    with pytest.raises(Exception):
        transcriber.transcribe_wav("/nonexistent/path.wav")