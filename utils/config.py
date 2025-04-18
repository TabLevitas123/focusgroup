from pathlib import Path
import json
import os
from utils.logger import get_logger

LOG = get_logger("Config")

CONFIG_PATH = Path.home() / ".focuspanel" / "config.json"
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

_DEFAULTS = {
    # Database and application settings
    "DB_PATH": str(Path.home() / ".focuspanel" / "focuspanel.db"),
    "SAMPLE_RATE": 16_000,
    
    # Model provider selection (openai, anthropic, google, huggingface, local)
    "MODEL_PROVIDER": "openai",
    "USE_AI_ENHANCED_RECOMMENDATIONS": False,
    
    # OpenAI settings
    "WHISPER_MODEL": "base",  # Changed to local base model as default
    "GPT_MODEL": "gpt-4o-mini",
    
    # Anthropic settings
    "CLAUDE_MODEL": "claude-3-haiku-20240307",
    
    # Google settings
    "GEMINI_MODEL": "gemini-1.5-pro",
    
    # Hugging Face settings
    "HF_TRANSCRIPTION_MODEL": "openai/whisper-base",
    "HF_SENTIMENT_MODEL": "distilbert-base-uncased-finetuned-sst-2-english",
    "HF_SUMMARIZATION_MODEL": "facebook/bart-large-cnn",
    
    # Local model settings
    "LOCAL_TRANSCRIPTION_MODEL": "",  # Path to local transcription model
    "LOCAL_LLM_MODEL": "",            # Path to local LLM model (e.g., LLaMA)
    "LOCAL_LLM_SERVER": "http://localhost:8000",  # URL for local LLM server
}

def load() -> dict:
    # Create a new config dictionary starting with defaults
    config_dict = _DEFAULTS.copy()
    
    # Add environment variables (so they're refreshed each time load() is called)
    # API keys for various services
    config_dict["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    config_dict["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")
    config_dict["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
    config_dict["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                config_dict.update(data)
        except json.JSONDecodeError as e:
            LOG.warning("Config is not valid JSON. Using defaults. %s", e)
        except Exception as e:
            LOG.warning("Failed to load config: %s", e)
    # Check for keys based on selected provider
    provider = config_dict.get("MODEL_PROVIDER", "openai")
    if provider == "openai" and not config_dict["OPENAI_API_KEY"]:
        LOG.warning("No OpenAI API key set. Add it to config or environment.")
    elif provider == "anthropic" and not config_dict["ANTHROPIC_API_KEY"]:
        LOG.warning("No Anthropic API key set. Add it to config or environment.")
    elif provider == "google" and not config_dict["GOOGLE_API_KEY"]:
        LOG.warning("No Google API key set. Add it to config or environment.")
    elif provider == "huggingface" and not config_dict["HF_TOKEN"]:
        LOG.warning("No Hugging Face token set. Some models may be unavailable.")
    elif provider == "local":
        if not config_dict["LOCAL_TRANSCRIPTION_MODEL"] and not config_dict["LOCAL_LLM_MODEL"]:
            LOG.warning("No local model paths configured. Check config file.")
    return config_dict

def save(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as e:
        LOG.error("Failed to save config: %s", e)

CFG = load()