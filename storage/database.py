import sqlite3, json, os
from utils.config import CFG
from utils.logger import get_logger

LOG = get_logger("Database")

def get_db_path():
    """Get the database path from environment or config."""
    return os.environ.get("DB_PATH", CFG.get("DB_PATH"))

def init():
    try:
        db_path = get_db_path()
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    audio_path TEXT,
                    profiles_json TEXT,
                    recommendations_json TEXT
                )
            """)
        LOG.info("Database initialized.")
    except Exception as e:
        LOG.error("DB init failed: %s", e)

def save_session(audio_path: str, profiles: list[dict], recommendations: dict):
    try:
        db_path = get_db_path()
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                INSERT INTO sessions (audio_path, profiles_json, recommendations_json)
                VALUES (?, ?, ?)
            """, (audio_path, json.dumps(profiles), json.dumps(recommendations)))
        LOG.info("Session saved to DB.")
    except Exception as e:
        LOG.error("DB insert failed: %s", e)