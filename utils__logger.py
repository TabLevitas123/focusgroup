import logging, sys, pathlib, os

LOG_PATH = pathlib.Path.home() / ".focuspanel" / "focuspanel.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

try:
    handlers = [
        logging.FileHandler(LOG_PATH, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
except Exception as e:
    handlers = [logging.StreamHandler(sys.stdout)]
    print(f"WARNING: Could not initialize log file: {e}")

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=handlers,
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)