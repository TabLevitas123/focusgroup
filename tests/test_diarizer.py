import pytest
import sys
from unittest.mock import patch

# Create a mock module for pyannote.audio
class MockPipeline:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()
    
    def __call__(self, path):
        if path == "/bad/path.wav":
            raise FileNotFoundError("No such file")
        return None

# Create mock modules
sys.modules['pyannote'] = type('pyannote', (), {})
sys.modules['pyannote.audio'] = type('audio', (), {'Pipeline': MockPipeline})
sys.modules['torch'] = type('torch', (), {})

# Now import the module under test
from processing import diarizer

def test_diarize_handles_invalid_file():
    with pytest.raises(Exception):
        diarizer.diarize("/bad/path.wav")