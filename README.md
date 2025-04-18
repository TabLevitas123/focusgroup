# FocusPanel

Focus Panel is a tool for analyzing focus group audio recordings. It processes audio recordings, transcribes the speech, identifies speakers, builds profile information, and provides recommendations based on the discussion.

## What's New: Multi-Model Support

FocusPanel now supports multiple AI model providers for transcription and analysis:

- **OpenAI** (Whisper and GPT models)
- **Anthropic** (Claude models)
- **Google** (Gemini models)
- **Hugging Face** (open-source models)
- **Local Models** (including LLaMA and other locally hosted models)

This flexibility allows you to choose the best model for your needs or use models that align with your privacy and budget requirements.

## Key Components

- **audio**: Audio recording functionality
- **gui**: User interface for recording, viewing and exporting session data
- **processing**:
  - Diarization (speaker identification)
  - Transcription (multi-model support)
  - Profile building (multi-model support)
  - Recommendation generation (multi-model support)
- **storage**: Database interactions for saving and retrieving sessions
- **utils**: Configuration and logging utilities

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure your preferred model provider (optional):
   ```
   # Create a configuration file
   mkdir -p ~/.focuspanel
   echo '{
     "MODEL_PROVIDER": "openai",
     "OPENAI_API_KEY": "your_api_key_here"
   }' > ~/.focuspanel/config.json
   ```

3. Run the application:
   ```
   python -m gui.main_window
   ```
   
   Alternatively, use the launcher script:
   ```
   ./launcher.sh
   ```

## Model Configuration

### OpenAI (Default)
```json
{
  "MODEL_PROVIDER": "openai",
  "OPENAI_API_KEY": "your_api_key_here",
  "WHISPER_MODEL": "base",
  "GPT_MODEL": "gpt-4o-mini"
}
```

### Anthropic (Claude)
```json
{
  "MODEL_PROVIDER": "anthropic",
  "ANTHROPIC_API_KEY": "your_api_key_here",
  "CLAUDE_MODEL": "claude-3-haiku-20240307"
}
```

### Google (Gemini)
```json
{
  "MODEL_PROVIDER": "google",
  "GOOGLE_API_KEY": "your_api_key_here",
  "GEMINI_MODEL": "gemini-1.5-pro"
}
```

### Hugging Face
```json
{
  "MODEL_PROVIDER": "huggingface",
  "HF_TOKEN": "your_token_here",
  "HF_TRANSCRIPTION_MODEL": "openai/whisper-base",
  "HF_SENTIMENT_MODEL": "distilbert-base-uncased-finetuned-sst-2-english",
  "HF_SUMMARIZATION_MODEL": "facebook/bart-large-cnn"
}
```

### Local Models
```json
{
  "MODEL_PROVIDER": "local",
  "LOCAL_TRANSCRIPTION_MODEL": "/path/to/whisper/model",
  "LOCAL_LLM_MODEL": "/path/to/llama/model.gguf",
  "LOCAL_LLM_SERVER": "http://localhost:8000"
}
```

## Environment Variables

API keys can also be set via environment variables:
```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export GOOGLE_API_KEY="your_google_key"
export HF_TOKEN="your_huggingface_token"
```

## Development Notes

- GUI tests require OpenGL libraries and are skipped in headless environments
- For database operations, the DB_PATH can be set via environment variable
- Log levels can be adjusted using the LOG_LEVEL environment variable
- Set USE_AI_ENHANCED_RECOMMENDATIONS to true in config to enable AI-powered recommendations

## Testing

Run the mock GUI for testing without real audio hardware:
```
python run_gui_mock.py
```

Run the headless test to verify all components in environments without GUI support:
```
python run_headless_test.py
```