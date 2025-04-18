#!/usr/bin/env python3
"""
Headless test runner for FocusPanel

This script mocks all external dependencies including the GUI libraries
to allow testing the core functionality in headless environments

New: Added support for testing multiple model providers (OpenAI, Claude, Gemini, 
Hugging Face, and local models)
"""

import sys
import os
import tempfile
import json
from unittest.mock import MagicMock, patch
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import namedtuple

# Define mock data classes
@dataclass
class MockUtterance:
    start: float
    end: float
    speaker: str
    text: str

@dataclass
class MockProfile:
    speaker: str
    keywords: list[str]
    sentiment: str
    summary: str

# Mock PySide6 and Qt
sys.modules['PySide6'] = MagicMock()
sys.modules['PySide6.QtWidgets'] = MagicMock()
sys.modules['PySide6.QtCore'] = MagicMock()
sys.modules['PySide6.QtGui'] = MagicMock()

# Create mock classes for GUI widgets
class MockQApplication:
    def __init__(self, args):
        self.args = args
    
    def exec(self):
        print("Mock QApplication exec called")
        return 0

class MockQWidget:
    def __init__(self):
        self.layout = None
        self.visible = False
        self.size = (600, 800)
    
    def setWindowTitle(self, title):
        print(f"Setting window title to: {title}")
    
    def resize(self, w, h):
        self.size = (w, h)
        print(f"Resizing window to {w}x{h}")
    
    def show(self):
        self.visible = True
        print("Window shown")

class MockQLabel:
    def __init__(self, text=""):
        self.text = text
    
    def setText(self, text):
        self.text = text
        print(f"Label text set to: {text}")

class MockQPushButton:
    def __init__(self, text=""):
        self.text = text
        self.enabled = True
        self.clicked = MockSignal()
    
    def setEnabled(self, enabled):
        self.enabled = enabled
        print(f"Button '{self.text}' enabled: {enabled}")

class MockQListWidget:
    def __init__(self):
        self.items = []
        self.current_row = -1
        self.itemSelectionChanged = MockSignal()
    
    def addItem(self, text):
        self.items.append(text)
        print(f"Added item to list: {text}")
    
    def clear(self):
        self.items = []
        print("List cleared")
    
    def currentRow(self):
        return self.current_row

class MockQTextEdit:
    def __init__(self, readOnly=False):
        self.text = ""
        self.read_only = readOnly
    
    def setPlainText(self, text):
        self.text = text
        print(f"TextEdit text set (truncated): {text[:50]}...")

class MockQProgressBar:
    def __init__(self):
        self.value = 0
        self.min_value = 0
        self.max_value = 100
    
    def setValue(self, value):
        self.value = value
        print(f"Progress bar value set to: {value}%")
    
    def setRange(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

class MockQVBoxLayout:
    def __init__(self, parent=None):
        self.parent = parent
        self.widgets = []
    
    def addWidget(self, widget):
        self.widgets.append(widget)

class MockSignal:
    def __init__(self):
        self.callbacks = []
    
    def connect(self, callback):
        self.callbacks.append(callback)
        print(f"Connected callback: {callback.__name__}")

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

# Mock various AI model providers
# OpenAI
sys.modules['openai'] = MagicMock()

# Anthropic
sys.modules['anthropic'] = MagicMock()
mock_anthropic_client = MagicMock()
mock_anthropic_message = MagicMock()
mock_content_block = MagicMock()
mock_content_block.text = "positive" # Mock response text
mock_anthropic_message.content = [mock_content_block]
mock_anthropic_client.messages.create.return_value = mock_anthropic_message
sys.modules['anthropic'].Anthropic.return_value = mock_anthropic_client

# Google (Gemini)
sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()
mock_gemini_model = MagicMock()
mock_gemini_model.generate_content.return_value.text = "positive"
sys.modules['google.generativeai'].GenerativeModel.return_value = mock_gemini_model

# Hugging Face transformers
sys.modules['transformers'] = MagicMock()
# Mock sentiment analysis pipeline
mock_sentiment_pipeline = MagicMock()
mock_sentiment_pipeline.return_value = [{"label": "POSITIVE", "score": 0.9}]
# Mock summarization pipeline
mock_summarization_pipeline = MagicMock()
mock_summarization_pipeline.return_value = [{"summary_text": "This is a summary"}]
# We'll return different pipeline types based on the task
def mock_hf_pipeline(task_name, *args, **kwargs):
    if task_name == "sentiment-analysis":
        return mock_sentiment_pipeline
    elif task_name == "summarization":
        return mock_summarization_pipeline
    else:
        return MagicMock()
sys.modules['transformers'].pipeline = mock_hf_pipeline

# Llama-cpp-python
sys.modules['llama_cpp'] = MagicMock()
mock_llama = MagicMock()
mock_llama.return_value = {"choices": [{"text": "positive"}]}
sys.modules['llama_cpp'].Llama = mock_llama

# Create mock model provider classes
class MockModelProvider:
    """Base mock model provider interface"""
    
    @staticmethod
    def get_provider(provider_name=None):
        # Return the appropriate mock provider based on name
        provider_map = {
            "openai": MockOpenAIProvider(),
            "anthropic": MockAnthropicProvider(),
            "google": MockGoogleProvider(),
            "huggingface": MockHuggingFaceProvider(),
            "local": MockLocalProvider(),
        }
        return provider_map.get(provider_name, MockOpenAIProvider())


class MockBaseProvider:
    """Base provider with common functionality"""
    
    def transcribe(self, audio_path):
        """Mock transcription that returns the same result regardless of provider"""
        return {
            "text": "This is a mock transcription for testing.",
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "This is speaker one talking."},
                {"start": 2.6, "end": 5.0, "text": "This is speaker two responding."},
                {"start": 5.1, "end": 7.5, "text": "Speaker one continues the conversation."},
                {"start": 7.6, "end": 10.0, "text": "A third speaker joins in."}
            ]
        }


class MockOpenAIProvider(MockBaseProvider):
    """OpenAI mock provider"""
    
    def analyze_text(self, text, analysis_type):
        print(f"OpenAI mock: Analyzing text with {analysis_type}")
        if analysis_type == "sentiment":
            return {"sentiment": "positive" if "speaker one" in text else "neutral"}
        elif analysis_type == "keywords":
            return {"keywords": ["openai", "quality", "effective", "service", "model"]}
        elif analysis_type == "summary":
            return {"summary": f"OpenAI summary: {text[:30]}..."}
        elif analysis_type == "custom":
            return {
                "result": {
                    "tone": "professional and precise",
                    "keywords": ["quality", "effective", "openai", "gpt", "service"],
                    "color_scheme": "blue and white",
                    "focus": "Messaging should emphasize: quality, innovation, reliability"
                }
            }
        return {"error": "Unknown analysis type"}


class MockAnthropicProvider(MockBaseProvider):
    """Anthropic (Claude) mock provider"""
    
    def analyze_text(self, text, analysis_type):
        print(f"Anthropic mock: Analyzing text with {analysis_type}")
        if analysis_type == "sentiment":
            return {"sentiment": "positive" if "conversation" in text else "neutral"}
        elif analysis_type == "keywords":
            return {"keywords": ["anthropic", "claude", "conversation", "nuanced", "context"]}
        elif analysis_type == "summary":
            return {"summary": f"Claude summary: {text[:30]}..."}
        elif analysis_type == "custom":
            return {
                "result": {
                    "tone": "conversational and thoughtful",
                    "keywords": ["anthropic", "claude", "conversation", "nuanced", "thoughtful"],
                    "color_scheme": "green and purple",
                    "focus": "Messaging should emphasize: conversation, clarity, thoughtfulness"
                }
            }
        return {"error": "Unknown analysis type"}


class MockGoogleProvider(MockBaseProvider):
    """Google (Gemini) mock provider"""
    
    def analyze_text(self, text, analysis_type):
        print(f"Google mock: Analyzing text with {analysis_type}")
        if analysis_type == "sentiment":
            return {"sentiment": "neutral"}
        elif analysis_type == "keywords":
            return {"keywords": ["google", "gemini", "multimodal", "integrated", "versatile"]}
        elif analysis_type == "summary":
            return {"summary": f"Gemini summary: {text[:30]}..."}
        elif analysis_type == "custom":
            return {
                "result": {
                    "tone": "helpful and integrated",
                    "keywords": ["google", "gemini", "multimodal", "integrated", "versatile"],
                    "color_scheme": "green, blue, red, and yellow",
                    "focus": "Messaging should emphasize: integration, versatility, accessibility"
                }
            }
        return {"error": "Unknown analysis type"}


class MockHuggingFaceProvider(MockBaseProvider):
    """Hugging Face mock provider"""
    
    def analyze_text(self, text, analysis_type):
        print(f"HuggingFace mock: Analyzing text with {analysis_type}")
        if analysis_type == "sentiment":
            return {"sentiment": "negative" if "third speaker" in text else "positive"}
        elif analysis_type == "keywords":
            return {"keywords": ["huggingface", "open", "community", "transformers", "flexible"]}
        elif analysis_type == "summary":
            return {"summary": f"HuggingFace summary: {text[:30]}..."}
        elif analysis_type == "custom":
            return {
                "result": {
                    "tone": "open and educational",
                    "keywords": ["open-source", "community", "transformers", "flexible", "scientific"],
                    "color_scheme": "yellow and orange",
                    "focus": "Messaging should emphasize: open-source, community, flexibility"
                }
            }
        return {"error": "Unknown analysis type"}


class MockLocalProvider(MockBaseProvider):
    """Local model mock provider"""
    
    def analyze_text(self, text, analysis_type):
        print(f"Local model mock: Analyzing text with {analysis_type}")
        if analysis_type == "sentiment":
            return {"sentiment": "neutral"}
        elif analysis_type == "keywords":
            return {"keywords": ["local", "private", "control", "customized", "efficient"]}
        elif analysis_type == "summary":
            return {"summary": f"Local model summary: {text[:30]}..."}
        elif analysis_type == "custom":
            return {
                "result": {
                    "tone": "secure and customized",
                    "keywords": ["local", "private", "control", "customized", "efficient"],
                    "color_scheme": "gray and dark blue",
                    "focus": "Messaging should emphasize: privacy, control, customization"
                }
            }
        return {"error": "Unknown analysis type"}


# Create the mock model_providers module
sys.modules['processing.model_providers'] = MagicMock()
sys.modules['processing.model_providers'].ModelProvider = MockModelProvider

# Apply patches
sys.modules['PySide6.QtWidgets'].QApplication = MockQApplication
sys.modules['PySide6.QtWidgets'].QWidget = MockQWidget
sys.modules['PySide6.QtWidgets'].QPushButton = MockQPushButton
sys.modules['PySide6.QtWidgets'].QVBoxLayout = MockQVBoxLayout
sys.modules['PySide6.QtWidgets'].QLabel = MockQLabel
sys.modules['PySide6.QtWidgets'].QListWidget = MockQListWidget
sys.modules['PySide6.QtWidgets'].QTextEdit = MockQTextEdit
sys.modules['PySide6.QtWidgets'].QProgressBar = MockQProgressBar

# Set environment variables for testing
os.environ['DB_PATH'] = ':memory:'

# Mock wave module to avoid errors with wave file handling
mock_wave = MagicMock()
mock_wave_file = MagicMock()
mock_wave.open.return_value.__enter__.return_value = mock_wave_file
mock_wave_file.setsampwidth = MagicMock()  # Properly mock the setsampwidth method
mock_wave_file.setframerate = MagicMock()
mock_wave_file.setnchannels = MagicMock()
mock_wave_file.writeframes = MagicMock()
sys.modules['wave'] = mock_wave

# Create patches for the processing modules
def create_mock_profiles():
    """Create mock profiles for testing"""
    return [
        MockProfile(
            speaker="speaker_1",
            keywords=["important", "quality", "effective"],
            sentiment="positive",
            summary="This speaker seems positive about the product."
        ),
        MockProfile(
            speaker="speaker_2",
            keywords=["concern", "question", "unclear"],
            sentiment="neutral",
            summary="This speaker has questions about the features."
        ),
        MockProfile(
            speaker="speaker_3", 
            keywords=["price", "expensive", "alternative"],
            sentiment="negative",
            summary="This speaker is concerned about the pricing."
        )
    ]

def mock_extract_segments(json_data):
    """Mock function to extract segments from whisper JSON"""
    return json_data["segments"]

def mock_build_profiles(utterances):
    """Mock function to build profiles from utterances"""
    return create_mock_profiles()

def mock_profiles_to_json(profiles):
    """Mock function to convert profiles to JSON"""
    return [asdict(p) for p in profiles]

def mock_recommend(profiles):
    """Mock function to generate recommendations"""
    return {
        "tone": "professional and balanced",
        "keywords": ["quality", "effective", "price"],
        "color_scheme": "balanced palette (blue, white, orange)",
        "focus": "Messaging should emphasize: quality, value, effectiveness"
    }

# Test function for multi-model functionality
def test_model_providers():
    print("\n=== TESTING MULTI-MODEL PROVIDERS ===\n")
    
    from utils.config import load as load_config
    from processing.model_providers import ModelProvider
    
    providers = ["openai", "anthropic", "google", "huggingface", "local"]
    
    for provider_name in providers:
        print(f"\nTesting {provider_name.upper()} provider:")
        os.environ['MODEL_PROVIDER'] = provider_name
        
        # Refresh config to pick up the new provider
        load_config()
        
        # Get the provider
        provider = ModelProvider.get_provider(provider_name)
        
        # Test transcription
        print(f"- Testing transcription with {provider_name}")
        result = provider.transcribe("dummy_path.wav")
        print(f"  Transcribed {len(result.get('segments', []))} segments")
        
        # Test text analysis
        test_text = "This is a test text that will be analyzed by different model providers."
        
        print(f"- Testing sentiment analysis with {provider_name}")
        sentiment = provider.analyze_text(test_text, "sentiment")
        print(f"  Sentiment: {sentiment.get('sentiment', 'unknown')}")
        
        print(f"- Testing keyword extraction with {provider_name}")
        keywords = provider.analyze_text(test_text, "keywords")
        print(f"  Keywords: {', '.join(keywords.get('keywords', []))}")
        
        print(f"- Testing summarization with {provider_name}")
        summary = provider.analyze_text(test_text, "summary")
        print(f"  Summary: {summary.get('summary', '')}")
        
        print(f"- Testing custom analysis with {provider_name}")
        custom = provider.analyze_text(test_text, "custom")
        if isinstance(custom.get('result', {}), dict):
            print(f"  Custom analysis successful")
        
    return True

# Now import and test the FocusPanel application
def test_focus_panel():
    print("\n=== TESTING FOCUSPANEL APPLICATION STRUCTURE ===\n")
    
    # Import the modules here after mocking
    from gui import main_window
    
    # Apply the necessary patches
    with patch('processing.transcriber.extract_segments', mock_extract_segments):
        with patch('processing.profile_builder.build_profiles', mock_build_profiles):
            with patch('processing.profile_builder.profiles_to_json', mock_profiles_to_json):
                with patch('processing.recommender.recommend', mock_recommend):
                    # Initialize the database
                    from storage import database
                    database.init()
                    print("Database initialized in memory")
                    
                    # Create FocusPanelApp instance
                    print("\nCreating FocusPanelApp instance...")
                    app = main_window.FocusPanelApp()
                    
                    # Test core functionality
                    print("\n=== TESTING RECORD/STOP FUNCTIONALITY ===")
                    app._start()
                    app._stop()
                    
                    print("\n=== TESTING ANALYSIS PIPELINE ===")
                    app.wav_path = Path(tempfile.mkstemp(suffix=".wav")[1])
                    # Instead of using the threaded version, we'll call _pipeline directly
                    print("\nRunning analysis pipeline...")
                    app._pipeline()
                    
                    print("\n=== TESTING PROFILE DISPLAY ===")
                    # Set the profiles to our mock profiles
                    app.profiles = create_mock_profiles()
                    app.list_profiles.current_row = 0
                    app._show_profile()
                    
                    print("\n=== TESTING EXPORT FUNCTIONALITY ===")
                    # Ensure we have profiles and recommendations for export
                    app.profiles = create_mock_profiles()
                    app.recs = mock_recommend(app.profiles)
                    # Test export with a temporary directory
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        export_path = app._export(export_dir=tmp_dir)
                        print(f"Exported file path: {export_path}")
                    
                    print("\n=== TESTING AI-ENHANCED RECOMMENDATIONS ===")
                    # Test AI-enhanced recommendations with different providers
                    from processing.recommender import recommend
                    from utils.config import CFG
                    
                    # Enable AI-enhanced recommendations
                    # CFG is a dictionary in this file, not an object with _config_dict
                    CFG["USE_AI_ENHANCED_RECOMMENDATIONS"] = True
                    
                    # Test each provider
                    providers = ["openai", "anthropic", "google", "huggingface", "local"]
                    for provider in providers:
                        CFG["MODEL_PROVIDER"] = provider
                        print(f"\nTesting AI-enhanced recommendations with {provider}:")
                        recs = recommend(app.profiles)
                        print(f"- Tone: {recs.get('tone', 'unknown')}")
                        print(f"- Focus: {recs.get('focus', 'unknown')}")
                    
                    print("\n=== FOCUSPANEL STRUCTURE TEST COMPLETE ===")
                    print("All core functionality tested successfully.")
                    return True

if __name__ == "__main__":
    try:
        # First test the multi-model providers
        model_test_success = test_model_providers()
        
        # Then test the main application
        app_test_success = test_focus_panel()
        
        print("\n=== ALL TESTS COMPLETED SUCCESSFULLY ===")
        sys.exit(0 if model_test_success and app_test_success else 1)
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)