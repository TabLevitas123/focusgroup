from pathlib import Path
import json
import os
from utils.logger import get_logger

LOG = get_logger("Config")

CONFIG_PATH = Path.home() / ".focuspanel" / "config.json"
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

_DEFAULTS = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "WHISPER_MODEL": "whisper-1",
    "GPT_MODEL": "gpt-4o-mini",
    "DB_PATH": str(Path.home() / ".focuspanel" / "focuspanel.db"),
    "SAMPLE_RATE": 16_000,
}

def load() -> dict:
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                _DEFAULTS.update(data)
        except json.JSONDecodeError as e:
            LOG.warning("Config is not valid JSON. Using defaults. %s", e)
        except Exception as e:
            LOG.warning("Failed to load config: %s", e)
    if not _DEFAULTS["OPENAI_API_KEY"]:
        LOG.warning("No OpenAI API key set. Add it to config or environment.")
    return _DEFAULTS

def save(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as e:
        LOG.error("Failed to save config: %s", e)

CFG = load()