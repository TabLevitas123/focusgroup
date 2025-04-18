from storage import database
import os

def test_init_and_insert(tmp_path):
    db_path = tmp_path / "test.db"
    os.environ["DB_PATH"] = str(db_path)
    database.init()
    database.save_session("audio.wav", [{"speaker": "S1"}], {"tone": "positive"})
    assert db_path.exists()