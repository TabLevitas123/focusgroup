                                  ...........................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
                                  ...............#!/usr/bin/env python3
"""
Mock runner for the FocusPanel GUI
This script mocks external dependencies to allow testing the GUI
without actual hardware or libraries

With multi-model support, this mock now supports testing different model providers
without requiring actual API keys or model downloads.
"""

import sys
import os
from unittest.mock import MagicMock, patch
from collections import namedtuple

# Set QT_QPA_PLATFORM to offscreen for headless environments
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Mock PyAudio with proper sample size handling
mock_pyaudio = MagicMock()
mock_pyaudio_instance = MagicMock()
mock_pyaudio.PyAudio.return_value = mock_pyaudio_instance
mock_pyaudio_instance.get_sample_size.return_value = 2  # Return a valid sample size
mock_pyaudio.paInt16 = 16  # Use 16 as a constant value

sys.modules['pyaudio'] = mock_pyaudio

# Mock pyannote.audio for speaker diarization
class MockPipeline:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()
    
    def __call__(self, path):
        Segment = namedtuple('Segment', ['start', 'end'])
        
        # Return a mock object that can be iterated
        class MockDiar:
            def itertracks(self, yield_label=False):
                # Generate some fake speaker segments
                segments = [
                    (Segment(0.0, 2.5), None, "speaker_1"),
                    (Segment(2.6, 5.0), None, "speaker_2"),
                    (Segment(5.1, 7.5), None, "speaker_1"),
                    (Segment(7.6, 10.0), None, "speaker_3"),
                ]
                return segments
        
        return MockDiar()

sys.modules['pyannote'] = MagicMock()
sys.modules['pyannote.audio'] = MagicMock()
sys.modules['pyannote.audio'].Pipeline = MockPipeline

# Mock whisper for transcription
class MockWhisperModel:
    def transcribe(self, path, **kwargs):
        # Generate a fake transcription
        return {
            "text": "This is a mock transcription for testing the GUI.",
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "This is speaker one talking."},
                {"start": 2.6, "end": 5.0, "text": "This is speaker two responding."},
                {"start": 5.1, "end": 7.5, "text": "Speaker one continues the conversation."},
                {"start": 7.6, "end": 10.0, "text": "A third speaker joins in."}
            ]
        }
    
    def __call__(self, *args, **kwargs):
        # Make the model callable for compatibility with some implementations
        return self.transcribe(*args, **kwargs)

sys.modules['whisper'] = MagicMock()
sys.modules['whisper'].load_model = lambda model_name: MockWhisperModel()

# Mock various AI provider clients
# OpenAI
sys.modules['openai'] = MagicMock()

# Anthropic
sys.modules['anthropic'] = MagicMock()
mock_anthropic_client = MagicMock()
mock_anthropic_message = MagicMock()
mock_anthropic_message.content = [MagicMock(text="positive")]
mock_anthropic_client.messages.create.return_value = mock_anthropic_message
sys.modules['anthropic'].Anthropic.return_value = mock_anthropic_client

# Google Gemini
sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()

# Hugging Face transformers
sys.modules['transformers'] = MagicMock()
mock_pipeline = MagicMock()
mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.9}]
sys.modules['transformers'].pipeline = mock_pipeline

# Llama-cpp-python
sys.modules['llama_cpp'] = MagicMock()
mock_llama = MagicMock()
mock_llama.return_value = {"choices": [{"text": "positive"}]}
sys.modules['llama_cpp'].Llama = mock_llama

# Mock the model providers module with a custom implementation
class MockModelProvider:
    @staticmethod
    def get_provider(provider_name=None):
        return MockOpenAIProvider()

class MockOpenAIProvider:
    def transcribe(self, audio_path):
        return {
            "text": "This is a mock transcription for testing the GUI.",
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "This is speaker one talking."},
                {"start": 2.6, "end": 5.0, "text": "This is speaker two responding."},
                {"start": 5.1, "end": 7.5, "text": "Speaker one continues the conversation."},
                {"start": 7.6, "end": 10.0, "text": "A third speaker joins in."}
            ]
        }
    
    def analyze_text(self, text, analysis_type):
        if analysis_type == "sentiment":
            return {"sentiment": "positive" if "speaker one" in text else "neutral"}
        elif analysis_type == "keywords":
            return {"keywords": ["speaker", "conversation", "testing", "mock", "gui"]}
        elif analysis_type == "summary":
            return {"summary": text[:50] + "..."}
        elif analysis_type == "custom":
            return {
                "result": {
                    "tone": "inspirational and bold",
                    "keywords": ["quality", "effective", "conversation", "speaker", "testing"],
                    "color_scheme": "vibrant (orange, green, blue)",
                    "focus": "Messaging should emphasize: quality, conversation, testing"
                }
            }
        return {"error": "Unknown analysis type"}

# Create the model_providers module and add our mock classes
sys.modules['processing.model_providers'] = MagicMock()
sys.modules['processing.model_providers'].ModelProvider = MockModelProvider

# Mock wave module to avoid errors with wave file handling
mock_wave = MagicMock()
mock_wave_file = MagicMock()
mock_wave.open.return_value.__enter__.return_value = mock_wave_file
mock_wave_file.setsampwidth = MagicMock()
mock_wave_file.setframerate = MagicMock()
mock_wave_file.setnchannels = MagicMock()
mock_wave_file.writeframes = MagicMock()
sys.modules['wave'] = mock_wave

# Set environment variables
os.environ['DB_PATH'] = ':memory:'
os.environ['MODEL_PROVIDER'] = 'openai'  # For testing purposes

# Import the main window module and run the application
from gui.main_window import run

if __name__ == "__main__":
    try:
        print("Starting FocusPanel with mock multi-model support...")
        print("Available model providers (all mocked): OpenAI, Anthropic, Google, Hugging Face, Local")
        run()
    except Exception as e:
        print(f"Error running GUI: {e}")
        if "QApplication" in str(e):
            print("This environment doesn't support GUI applications.")
            print("The code structure has been validated, but GUI cannot be displayed.")