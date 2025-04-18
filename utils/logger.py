import logging, sys, pathlib, os

LOG_PATH = pathlib.Path.home() / ".focuspanel" / "focuspanel.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Set up basic logging configuration once
try:
    handlers = [
        logging.FileHandler(LOG_PATH, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
except Exception as e:
    handlers = [logging.StreamHandler(sys.stdout)]
    print(f"WARNING: Could not initialize log file: {e}")

# Initialize with default format
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=handlers,
)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name, respecting current LOG_LEVEL environment variable."""
    logger = logging.getLogger(name)
    
    # Always check current environment for log level
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    return logger