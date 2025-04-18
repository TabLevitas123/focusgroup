from utils.logger import get_logger
import logging

def test_logger_instance():
    logger = get_logger("LoggerTest")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "LoggerTest"

def test_logger_stream_handler(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    logger = get_logger("LevelTest")
    assert logger.level == logging.DEBUG