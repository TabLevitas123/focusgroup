import tempfile
import sys
from unittest.mock import patch, MagicMock

# Create mock classes for pyaudio
class MockPyAudio:
    def open(self, **kwargs):
        return MagicMock()
    
    def get_sample_size(self, format_type):
        return 2
    
    def terminate(self):
        pass

# Mock the modules
sys.modules['pyaudio'] = MagicMock()
sys.modules['pyaudio'].PyAudio = MockPyAudio
sys.modules['pyaudio'].paInt16 = 8

# Now import the module under test
from audio.recorder import record_session

@patch('wave.open')
def test_record_short_session(mock_wave):
    # Set up the wave mock
    mock_wave_instance = MagicMock()
    mock_wave.return_value.__enter__.return_value = mock_wave_instance
    
    # Run the test
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    rec = record_session(tmp.name, duration=1.0)
    rec.join(timeout=2)
    
    # Assert the filename
    assert tmp.name.endswith(".wav")
    
    # Verify the wave file was created
    mock_wave.assert_called_once()