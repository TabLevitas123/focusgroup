import tempfile
from audio.recorder import record_session

def test_record_short_session():
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    rec = record_session(tmp.name, duration=1.0)
    rec.join(timeout=2)
    assert tmp.name.endswith(".wav")