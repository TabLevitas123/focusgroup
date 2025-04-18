from processing import diarizer
import pytest

def test_diarize_handles_invalid_file():
    with pytest.raises(Exception):
        diarizer.diarize("/bad/path.wav")