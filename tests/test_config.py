import json
import os
from utils import config
from pathlib import Path

def test_load_config_returns_defaults_when_missing(tmp_path, monkeypatch):
    fake_path = tmp_path / "nonexistent_config.json"
    monkeypatch.setattr(config, "CONFIG_PATH", fake_path)
    result = config.load()
    assert isinstance(result, dict)
    assert "OPENAI_API_KEY" in result

def test_load_config_uses_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    cfg = config.load()
    assert cfg["OPENAI_API_KEY"] == "env-key"

def test_save_and_load_custom_key(tmp_path, monkeypatch):
    path = tmp_path / "cfg.json"
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    config.save({"X": "Y"})
    loaded = config.load()
    assert loaded["X"] == "Y"

def test_load_with_invalid_json(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    monkeypatch.setattr(config, "CONFIG_PATH", bad)
    cfg = config.load()
    assert isinstance(cfg, dict)