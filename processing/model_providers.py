"""
Model provider architecture for FocusPanel

This module defines the provider interface and implementations for various
AI model providers that can be used for transcription and text analysis.

Supported providers:
- OpenAI (Whisper and GPT models)
- Anthropic (Claude models)
- Google (Gemini models)  
- Hugging Face (open-source models)
- Local (locally hosted models)
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from utils.logger import get_logger
from utils.config import CFG

LOG = get_logger("ModelProviders")

class ModelProvider:
    """
    Factory class for creating and managing model providers
    """
    _providers = {}
    
    @staticmethod
    def get_provider(provider_name: Optional[str] = None) -> 'Provider':
        """
        Get a model provider instance
        
        Args:
            provider_name: Name of the provider (openai, anthropic, google, huggingface, local)
                           If None, uses the provider from config
        
        Returns:
            Provider instance
        """
        if provider_name is None:
            provider_name = CFG.get("MODEL_PROVIDER", "openai")
            
        LOG.info(f"Getting provider: {provider_name}")
        
        # Initialize provider if not already done
        if provider_name not in ModelProvider._providers:
            provider_map = {
                "openai": OpenAIProvider,
                "anthropic": AnthropicProvider,
                "google": GoogleProvider,
                "huggingface": HuggingFaceProvider,
                "local": LocalProvider
            }
            
            if provider_name not in provider_map:
                LOG.warning(f"Unknown provider: {provider_name}, defaulting to OpenAI")
                provider_name = "openai"
                
            provider_class = provider_map[provider_name]
            ModelProvider._providers[provider_name] = provider_class()
            
        return ModelProvider._providers[provider_name]


class Provider(ABC):
    """
    Base class for all model providers
    """
    
    @abstractmethod
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio to text
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict with transcription text and segments
        """
        pass
    
    @abstractmethod
    def analyze_text(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """
        Analyze text for various purposes (sentiment, keywords, summary)
        
        Args:
            text: The text to analyze
            analysis_type: Type of analysis ("sentiment", "keywords", "summary", "custom")
            
        Returns:
            Dict with analysis results
        """
        pass


class OpenAIProvider(Provider):
    """
    Provider implementation for OpenAI (Whisper and GPT)
    """
    
    def __init__(self):
        self.api_key = CFG.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        self.whisper_model = CFG.get("WHISPER_MODEL", "base")
        self.gpt_model = CFG.get("GPT_MODEL", "gpt-4o-mini")
        
        if not self.api_key:
            LOG.warning("No OpenAI API key provided. Using local Whisper model.")
        
        # Import here to avoid circular imports
        import openai
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Import whisper for local model option
        try:
            import whisper
            self.whisper = whisper
        except ImportError:
            LOG.warning("Could not import whisper. Local transcription may not work.")
            self.whisper = None
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio using OpenAI Whisper
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict with transcription text and segments
        """
        LOG.info(f"Transcribing with OpenAI (model: {self.whisper_model})")
        
        try:
            # Use API if key is available and model is not "base" or "tiny"
            if self.api_key and self.whisper_model not in ["base", "tiny"]:
                # Use OpenAI API
                with open(audio_path, "rb") as audio_file:
                    LOG.info(f"Using OpenAI API for transcription")
                    response = self.client.audio.transcriptions.create(
                        file=audio_file,
                        model=self.whisper_model,
                        response_format="verbose_json",
                        language="en"
                    )
                    
                    # Convert to dict if it's a response object
                    if not isinstance(response, dict):
                        response = response.model_dump()
                    
                    return response
            else:
                # Use local whisper model
                LOG.info(f"Using local Whisper model for transcription: {self.whisper_model}")
                if not self.whisper:
                    raise ImportError("Whisper not available for local transcription")
                
                model = self.whisper.load_model(self.whisper_model)
                result = model.transcribe(audio_path, language="en")
                LOG.info(f"Local transcription completed with {len(result.get('segments', []))} segments")
                return result
                
        except Exception as e:
            LOG.error(f"Error during OpenAI transcription: {e}")
            # Return a minimal result in case of error
            return {
                "text": "Error during transcription. Please try again.",
                "segments": []
            }
    
    def analyze_text(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """
        Analyze text using OpenAI GPT
        
        Args:
            text: The text to analyze
            analysis_type: Type of analysis ("sentiment", "keywords", "summary", "custom")
            
        Returns:
            Dict with analysis results
        """
        LOG.info(f"Analyzing text with OpenAI (model: {self.gpt_model})")
        
        if not self.api_key:
            LOG.warning("No OpenAI API key. Using basic analysis.")
            return self._fallback_analysis(text, analysis_type)
        
        try:
            # Define system prompts for different analysis types
            prompts = {
                "sentiment": "Analyze the sentiment of the following text and respond with only one word: positive, negative, or neutral.",
                "keywords": "Extract 5 important keywords from the following text. Respond with only a JSON array of strings.",
                "summary": "Summarize the following text in one short paragraph (max 100 words).",
                "custom": "Analyze the following text and provide marketing recommendations."
            }
            
            system_prompt = prompts.get(analysis_type, prompts["custom"])
            
            # Handle custom analysis specially
            if analysis_type == "custom":
                response = self.client.chat.completions.create(
                    model=self.gpt_model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": """You are a marketing analyst. 
                         Analyze the provided text and create JSON with these fields:
                         tone: a marketing tone recommendation (string)
                         keywords: an array of 5 important keywords to emphasize (array of strings)
                         color_scheme: recommended color palette (string)
                         focus: key messaging focus (string)"""},
                        {"role": "user", "content": text}
                    ]
                )
                
                result_json = json.loads(response.choices[0].message.content)
                return {"result": result_json}
            
            # Handle standard analysis types
            response = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
            )
            
            result = response.choices[0].message.content.strip()
            
            # Format the response based on the analysis type
            if analysis_type == "sentiment":
                sentiment = result.lower()
                if sentiment not in ["positive", "negative", "neutral"]:
                    sentiment = "neutral"  # Default if not recognized
                return {"sentiment": sentiment}
                
            elif analysis_type == "keywords":
                try:
                    # Try to parse as JSON
                    keywords = json.loads(result)
                    if not isinstance(keywords, list):
                        keywords = result.split()  # Fallback
                        
                    # Ensure we have a list of strings
                    keywords = [str(k).strip() for k in keywords if k][:5]  # Limit to 5
                    return {"keywords": keywords}
                except:
                    # Fall back to splitting by commas or spaces
                    keywords = [k.strip() for k in result.replace(',', ' ').split()][:5]
                    return {"keywords": keywords}
                    
            elif analysis_type == "summary":
                return {"summary": result}
            
            else:
                return {"error": "Unknown analysis type"}
                
        except Exception as e:
            LOG.error(f"Error during OpenAI text analysis: {e}")
            return self._fallback_analysis(text, analysis_type)
    
    def _fallback_analysis(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """Fallback analysis when API calls fail"""
        if analysis_type == "sentiment":
            # Simple keyword-based sentiment
            positive_words = ["good", "great", "excellent", "amazing", "love", "like", "positive"]
            negative_words = ["bad", "terrible", "awful", "hate", "dislike", "negative"]
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return {"sentiment": "positive"}
            elif neg_count > pos_count:
                return {"sentiment": "negative"}
            else:
                return {"sentiment": "neutral"}
                
        elif analysis_type == "keywords":
            # Simple word frequency
            words = [w.strip('.,!?()[]{}:;"\'').lower() for w in text.split()]
            # Remove short words and duplicates, get top 5
            word_freq = {}
            for word in words:
                if len(word) > 4:  # Only consider words longer than 4 chars
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            keywords = [word for word, _ in top_words[:5]]
            return {"keywords": keywords}
            
        elif analysis_type == "summary":
            # Simple truncation summary
            return {"summary": text[:150] + "..." if len(text) > 150 else text}
            
        elif analysis_type == "custom":
            # Generic recommendations
            return {
                "result": {
                    "tone": "professional and concise",
                    "keywords": ["quality", "service", "value", "solution", "performance"],
                    "color_scheme": "blue and white (professional)",
                    "focus": "Messaging should emphasize: quality, value, and reliability"
                }
            }
            
        else:
            return {"error": "Unknown analysis type"}


class AnthropicProvider(Provider):
    """
    Provider implementation for Anthropic (Claude)
    """
    
    def __init__(self):
        self.api_key = CFG.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
        self.model = CFG.get("CLAUDE_MODEL", "claude-3-haiku-20240307")
        
        if not self.api_key:
            LOG.warning("No Anthropic API key provided. Basic functionality only.")
        
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            LOG.error("Could not import Anthropic Python library")
            self.client = None
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Anthropic doesn't provide audio transcription, so we fall back to Whisper
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict with transcription text and segments
        """
        LOG.info("Anthropic does not support transcription, falling back to local Whisper")
        
        # Use the OpenAI provider for transcription
        openai_provider = OpenAIProvider()
        return openai_provider.transcribe(audio_path)
    
    def analyze_text(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """
        Analyze text using Anthropic Claude
        
        Args:
            text: The text to analyze
            analysis_type: Type of analysis ("sentiment", "keywords", "summary", "custom")
            
        Returns:
            Dict with analysis results
        """
        LOG.info(f"Analyzing text with Anthropic Claude (model: {self.model})")
        
        if not self.api_key or not self.client:
            LOG.warning("No Anthropic API key or client. Using basic analysis.")
            # Fall back to OpenAI's implementation for basic analysis
            return OpenAIProvider()._fallback_analysis(text, analysis_type)
        
        try:
            # Define prompts for different analysis types
            prompts = {
                "sentiment": "Analyze the sentiment of the following text and respond with only one word: positive, negative, or neutral.\n\nText: {text}",
                "keywords": "Extract 5 important keywords from the following text. Respond with only a JSON array of strings.\n\nText: {text}",
                "summary": "Summarize the following text in one short paragraph (max 100 words).\n\nText: {text}",
                "custom": """Analyze the following text and provide marketing recommendations in this JSON format:
                {
                    "tone": "recommended tone",
                    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
                    "color_scheme": "recommended colors",
                    "focus": "key messaging focus"
                }
                
                Text: {text}"""
            }
            
            prompt = prompts.get(analysis_type, prompts["custom"]).format(text=text)
            
            # Call Claude API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Get the text from the response
            result = message.content[0].text
            
            # Format the response based on the analysis type
            if analysis_type == "sentiment":
                sentiment = result.lower().strip()
                if sentiment not in ["positive", "negative", "neutral"]:
                    sentiment = "neutral"  # Default if not recognized
                return {"sentiment": sentiment}
                
            elif analysis_type == "keywords":
                try:
                    # Try to parse as JSON
                    keywords = json.loads(result)
                    if not isinstance(keywords, list):
                        keywords = result.split()  # Fallback
                        
                    # Ensure we have a list of strings
                    keywords = [str(k).strip() for k in keywords if k][:5]  # Limit to 5
                    return {"keywords": keywords}
                except:
                    # Fall back to splitting by commas or spaces
                    keywords = [k.strip() for k in result.replace(',', ' ').split()][:5]
                    return {"keywords": keywords}
                    
            elif analysis_type == "summary":
                return {"summary": result}
                
            elif analysis_type == "custom":
                try:
                    # Try to extract JSON from the response
                    # Find the start and end of the JSON object
                    json_start = result.find('{')
                    json_end = result.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = result[json_start:json_end]
                        result_json = json.loads(json_str)
                        return {"result": result_json}
                    else:
                        # Fallback
                        return OpenAIProvider()._fallback_analysis(text, analysis_type)
                except Exception as json_err:
                    LOG.error(f"Error parsing Claude JSON: {json_err}")
                    return OpenAIProvider()._fallback_analysis(text, analysis_type)
            
            else:
                return {"error": "Unknown analysis type"}
                
        except Exception as e:
            LOG.error(f"Error during Claude text analysis: {e}")
            return OpenAIProvider()._fallback_analysis(text, analysis_type)


class GoogleProvider(Provider):
    """
    Provider implementation for Google (Gemini)
    """
    
    def __init__(self):
        self.api_key = CFG.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        self.model = CFG.get("GEMINI_MODEL", "gemini-1.5-pro")
        
        if not self.api_key:
            LOG.warning("No Google API key provided. Basic functionality only.")
        
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            self.genai = genai
            self.model_client = genai.GenerativeModel(self.model)
        except ImportError:
            LOG.error("Could not import Google GenerativeAI library")
            self.genai = None
            self.model_client = None
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Google Gemini doesn't provide audio transcription, so we fall back to Whisper
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict with transcription text and segments
        """
        LOG.info("Google does not support transcription through Gemini, falling back to local Whisper")
        
        # Use the OpenAI provider for transcription
        openai_provider = OpenAIProvider()
        return openai_provider.transcribe(audio_path)
    
    def analyze_text(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """
        Analyze text using Google Gemini
        
        Args:
            text: The text to analyze
            analysis_type: Type of analysis ("sentiment", "keywords", "summary", "custom")
            
        Returns:
            Dict with analysis results
        """
        LOG.info(f"Analyzing text with Google Gemini (model: {self.model})")
        
        if not self.api_key or not self.model_client:
            LOG.warning("No Google API key or client. Using basic analysis.")
            # Fall back to OpenAI's implementation for basic analysis
            return OpenAIProvider()._fallback_analysis(text, analysis_type)
        
        try:
            # Define prompts for different analysis types
            prompts = {
                "sentiment": "Analyze the sentiment of the following text and respond with only one word: positive, negative, or neutral.\n\nText: {text}",
                "keywords": "Extract 5 important keywords from the following text. Respond with only these 5 keywords separated by commas.\n\nText: {text}",
                "summary": "Summarize the following text in one short paragraph (max 100 words).\n\nText: {text}",
                "custom": """Analyze the following text and provide marketing recommendations in this JSON format:
                {
                    "tone": "recommended tone",
                    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
                    "color_scheme": "recommended colors",
                    "focus": "key messaging focus"
                }
                
                Text: {text}"""
            }
            
            prompt = prompts.get(analysis_type, prompts["custom"]).format(text=text)
            
            # Call Gemini API
            response = self.model_client.generate_content(prompt)
            result = response.text
            
            # Format the response based on the analysis type
            if analysis_type == "sentiment":
                sentiment = result.lower().strip()
                if sentiment not in ["positive", "negative", "neutral"]:
                    sentiment = "neutral"  # Default if not recognized
                return {"sentiment": sentiment}
                
            elif analysis_type == "keywords":
                # Split by commas and clean up
                keywords = [k.strip() for k in result.split(',')]
                # Keep only non-empty values and limit to 5
                keywords = [k for k in keywords if k][:5]
                return {"keywords": keywords}
                    
            elif analysis_type == "summary":
                return {"summary": result}
                
            elif analysis_type == "custom":
                try:
                    # Try to extract JSON from the response
                    # Find the start and end of the JSON object
                    json_start = result.find('{')
                    json_end = result.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = result[json_start:json_end]
                        result_json = json.loads(json_str)
                        return {"result": result_json}
                    else:
                        # Fallback
                        return OpenAIProvider()._fallback_analysis(text, analysis_type)
                except Exception as json_err:
                    LOG.error(f"Error parsing Gemini JSON: {json_err}")
                    return OpenAIProvider()._fallback_analysis(text, analysis_type)
            
            else:
                return {"error": "Unknown analysis type"}
                
        except Exception as e:
            LOG.error(f"Error during Gemini text analysis: {e}")
            return OpenAIProvider()._fallback_analysis(text, analysis_type)


class HuggingFaceProvider(Provider):
    """
    Provider implementation for Hugging Face models
    """
    
    def __init__(self):
        self.token = CFG.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
        self.transcription_model = CFG.get("HF_TRANSCRIPTION_MODEL", "openai/whisper-base")
        self.sentiment_model = CFG.get("HF_SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
        self.summarization_model = CFG.get("HF_SUMMARIZATION_MODEL", "facebook/bart-large-cnn")
        
        self.pipelines = {}
        
        try:
            from transformers import pipeline
            self.pipeline_fn = pipeline
        except ImportError:
            LOG.error("Could not import Hugging Face transformers library")
            self.pipeline_fn = None
    
    def _get_pipeline(self, task, model):
        """Get or create a pipeline for a specific task"""
        if self.pipeline_fn is None:
            return None
            
        cache_key = f"{task}_{model}"
        if cache_key not in self.pipelines:
            try:
                LOG.info(f"Creating HuggingFace pipeline for {task} using {model}")
                use_auth = bool(self.token)
                self.pipelines[cache_key] = self.pipeline_fn(
                    task=task, 
                    model=model,
                    token=self.token if use_auth else None
                )
            except Exception as e:
                LOG.error(f"Error creating pipeline {task} with {model}: {e}")
                return None
                
        return self.pipelines.get(cache_key)
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio using Hugging Face models
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict with transcription text and segments
        """
        LOG.info(f"Transcribing with HuggingFace (model: {self.transcription_model})")
        
        try:
            # Handle different models
            if "whisper" in self.transcription_model.lower():
                # Use OpenAI Whisper from HF or locally
                pipe = self._get_pipeline("automatic-speech-recognition", self.transcription_model)
                
                if pipe:
                    # Use the HF pipeline
                    result = pipe(audio_path, chunk_length_s=30, return_timestamps=True)
                    
                    # Convert to a format similar to Whisper's output
                    chunks = result.get("chunks", [])
                    
                    segments = []
                    for i, chunk in enumerate(chunks):
                        segments.append({
                            "start": chunk.get("timestamp", (0, 0))[0],
                            "end": chunk.get("timestamp", (0, 0))[1],
                            "text": chunk.get("text", "")
                        })
                    
                    return {
                        "text": result.get("text", ""),
                        "segments": segments
                    }
                else:
                    # Fall back to local Whisper
                    LOG.info("Falling back to local Whisper")
                    openai_provider = OpenAIProvider()
                    return openai_provider.transcribe(audio_path)
            else:
                # Generic ASR model from HuggingFace
                pipe = self._get_pipeline("automatic-speech-recognition", self.transcription_model)
                
                if pipe:
                    result = pipe(audio_path)
                    
                    if isinstance(result, dict):
                        text = result.get("text", "")
                    else:
                        text = str(result)
                    
                    # Basic segments with estimated timestamps
                    words = text.split()
                    avg_word_duration = 0.5  # Rough estimate
                    segments = []
                    
                    current_segment = []
                    segment_words = 10  # Group into segments of this many words
                    
                    for i, word in enumerate(words):
                        current_segment.append(word)
                        
                        if len(current_segment) >= segment_words or i == len(words) - 1:
                            segment_text = " ".join(current_segment)
                            start_time = (i - len(current_segment) + 1) * avg_word_duration
                            end_time = (i + 1) * avg_word_duration
                            
                            segments.append({
                                "start": start_time,
                                "end": end_time,
                                "text": segment_text
                            })
                            
                            current_segment = []
                    
                    return {
                        "text": text,
                        "segments": segments
                    }
                else:
                    # Fall back to Whisper
                    LOG.info("HuggingFace pipeline unavailable, falling back to OpenAI")
                    openai_provider = OpenAIProvider()
                    return openai_provider.transcribe(audio_path)
                    
        except Exception as e:
            LOG.error(f"Error during HuggingFace transcription: {e}")
            # Fall back to Whisper
            LOG.info("Error with HuggingFace, falling back to OpenAI")
            openai_provider = OpenAIProvider()
            return openai_provider.transcribe(audio_path)
    
    def analyze_text(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """
        Analyze text using Hugging Face models
        
        Args:
            text: The text to analyze
            analysis_type: Type of analysis ("sentiment", "keywords", "summary", "custom")
            
        Returns:
            Dict with analysis results
        """
        LOG.info(f"Analyzing text with HuggingFace")
        
        try:
            if analysis_type == "sentiment":
                # Use sentiment analysis model
                pipe = self._get_pipeline("sentiment-analysis", self.sentiment_model)
                
                if pipe:
                    # HF models typically use labels like "POSITIVE"/"NEGATIVE"
                    result = pipe(text[:1024])  # Truncate to avoid token limits
                    
                    if isinstance(result, list) and len(result) > 0:
                        label = result[0].get("label", "NEUTRAL")
                        
                        if "POSITIVE" in label:
                            sentiment = "positive"
                        elif "NEGATIVE" in label:
                            sentiment = "negative"
                        else:
                            sentiment = "neutral"
                            
                        return {"sentiment": sentiment}
                    else:
                        return {"sentiment": "neutral"}
                else:
                    # Fall back to basic analysis
                    return OpenAIProvider()._fallback_analysis(text, analysis_type)
                    
            elif analysis_type == "keywords":
                # For keywords, we need a different approach
                # We could use:
                # 1. Keywords extraction models 
                # 2. NER models to extract entities
                # 3. Topic modeling
                
                # For simplicity, we'll use a keyword extraction model if available
                try:
                    from keybert import KeyBERT
                    kb = KeyBERT()
                    keywords = kb.extract_keywords(text, top_n=5)
                    return {"keywords": [k[0] for k in keywords]}
                except ImportError:
                    # Fall back to basic TF-IDF approach
                    return OpenAIProvider()._fallback_analysis(text, analysis_type)
                    
            elif analysis_type == "summary":
                # Use summarization model
                pipe = self._get_pipeline("summarization", self.summarization_model)
                
                if pipe:
                    # Limit input length to avoid OOM errors
                    max_length = 1024
                    truncated_text = text[:max_length]
                    
                    # Generate summary
                    result = pipe(truncated_text, max_length=100, min_length=30, do_sample=False)
                    
                    if isinstance(result, list) and len(result) > 0:
                        summary = result[0].get("summary_text", "")
                        return {"summary": summary}
                    else:
                        return {"summary": truncated_text[:150] + "..."}
                else:
                    # Fall back to basic analysis
                    return OpenAIProvider()._fallback_analysis(text, analysis_type)
                    
            elif analysis_type == "custom":
                # For custom analysis, we would need a text generation model
                # This is more complex with HuggingFace and requires a larger model
                
                # Fall back to basic recommendations
                LOG.info("Custom analysis not optimally supported with HuggingFace, using fallback")
                return OpenAIProvider()._fallback_analysis(text, "custom")
            
            else:
                return {"error": "Unknown analysis type"}
                
        except Exception as e:
            LOG.error(f"Error during HuggingFace text analysis: {e}")
            return OpenAIProvider()._fallback_analysis(text, analysis_type)


class LocalProvider(Provider):
    """
    Provider implementation for local models (e.g., LLaMA via llama-cpp-python)
    """
    
    def __init__(self):
        self.transcription_model = CFG.get("LOCAL_TRANSCRIPTION_MODEL", "")
        self.llm_model = CFG.get("LOCAL_LLM_MODEL", "")
        self.llm_server = CFG.get("LOCAL_LLM_SERVER", "http://localhost:8000")
        
        # Check if we have valid model paths
        if not self.transcription_model and not self.llm_model:
            LOG.warning("No local model paths configured. Some features may not work.")
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio using local Whisper
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict with transcription text and segments
        """
        LOG.info(f"Transcribing with local model: {self.transcription_model or 'default'}")
        
        try:
            # Check if we have a local transcription model path
            custom_model_path = None
            if self.transcription_model and Path(self.transcription_model).exists():
                custom_model_path = self.transcription_model
                
            # Import whisper
            try:
                import whisper
                
                # Load the model - either from custom path or use standard model
                if custom_model_path:
                    LOG.info(f"Loading custom model from {custom_model_path}")
                    model = whisper.load_model(custom_model_path)
                else:
                    LOG.info("Loading standard base Whisper model")
                    model = whisper.load_model("base")
                
                # Transcribe
                result = model.transcribe(audio_path, language="en")
                LOG.info(f"Transcription complete with {len(result.get('segments', []))} segments")
                return result
                
            except ImportError:
                LOG.error("Could not import Whisper. Falling back to OpenAI.")
                openai_provider = OpenAIProvider()
                return openai_provider.transcribe(audio_path)
                
        except Exception as e:
            LOG.error(f"Error during local transcription: {e}")
            # Fall back to OpenAI
            LOG.info("Error with local transcription, falling back to OpenAI")
            openai_provider = OpenAIProvider()
            return openai_provider.transcribe(audio_path)
    
    def analyze_text(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """
        Analyze text using local LLM
        
        Args:
            text: The text to analyze
            analysis_type: Type of analysis ("sentiment", "keywords", "summary", "custom")
            
        Returns:
            Dict with analysis results
        """
        LOG.info(f"Analyzing text with local LLM: {self.llm_model or 'server'}")
        
        try:
            # Check if we should use direct model loading or server
            if self.llm_model and Path(self.llm_model).exists():
                return self._analyze_with_local_model(text, analysis_type)
            elif self.llm_server:
                return self._analyze_with_server(text, analysis_type)
            else:
                LOG.warning("No local LLM configured. Using fallback analysis.")
                return OpenAIProvider()._fallback_analysis(text, analysis_type)
                
        except Exception as e:
            LOG.error(f"Error during local LLM analysis: {e}")
            return OpenAIProvider()._fallback_analysis(text, analysis_type)
    
    def _analyze_with_local_model(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """Use llama-cpp-python to analyze with a local model"""
        try:
            from llama_cpp import Llama
            
            # Load the model
            LOG.info(f"Loading LLaMA model from {self.llm_model}")
            llm = Llama(model_path=self.llm_model, n_ctx=2048)
            
            # Define prompts for different analysis types
            prompts = {
                "sentiment": f"Analyze the sentiment of the following text and respond with only one word: positive, negative, or neutral.\n\nText: {text}\n\nSentiment:",
                "keywords": f"Extract 5 important keywords from the following text and list them.\n\nText: {text}\n\nKeywords:",
                "summary": f"Summarize the following text in one short paragraph.\n\nText: {text}\n\nSummary:",
                "custom": f"""Analyze the following text and provide marketing recommendations.

Text: {text}

Provide recommendations in this structure:
Tone: 
Keywords: 
Color Scheme: 
Focus: """
            }
            
            prompt = prompts.get(analysis_type, prompts["custom"])
            
            # Generate response
            response = llm(
                prompt,
                max_tokens=256,
                temperature=0.1,
                echo=False
            )
            
            # Extract the generated text
            result = response.get("choices", [{}])[0].get("text", "").strip()
            
            # Process the result based on analysis type
            if analysis_type == "sentiment":
                result_lower = result.lower()
                if "positive" in result_lower:
                    return {"sentiment": "positive"}
                elif "negative" in result_lower:
                    return {"sentiment": "negative"}
                else:
                    return {"sentiment": "neutral"}
                    
            elif analysis_type == "keywords":
                # Extract keywords from the generated text
                # Split by commas, newlines, or spaces
                keywords = []
                for word in result.replace(',', ' ').replace('\n', ' ').split():
                    word = word.strip('.,!?()[]{};:"\'')
                    if word and len(word) > 2 and word.lower() not in ["and", "the", "for", "keywords", "keyword"]:
                        keywords.append(word)
                
                return {"keywords": keywords[:5]}  # Limit to 5 keywords
                
            elif analysis_type == "summary":
                return {"summary": result}
                
            elif analysis_type == "custom":
                # Try to extract structured information
                lines = result.split('\n')
                rec = {}
                
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        if key == "tone":
                            rec["tone"] = value
                        elif key == "keywords":
                            # Split keywords
                            keywords = [k.strip() for k in value.split(',')]
                            rec["keywords"] = [k for k in keywords if k][:5]
                        elif key == "color scheme":
                            rec["color_scheme"] = value
                        elif key == "focus":
                            rec["focus"] = value
                
                # Ensure we have all required fields
                if not all(k in rec for k in ["tone", "keywords", "color_scheme", "focus"]):
                    # Fall back to default values for missing fields
                    default = {
                        "tone": "professional and straightforward",
                        "keywords": ["quality", "value", "reliability", "performance", "solution"],
                        "color_scheme": "neutral and balanced",
                        "focus": "Messaging should emphasize quality and reliability"
                    }
                    
                    for k, v in default.items():
                        if k not in rec or not rec[k]:
                            rec[k] = v
                
                return {"result": rec}
            
            else:
                return {"error": "Unknown analysis type"}
                
        except ImportError:
            LOG.error("Could not import llama-cpp-python. Make sure it's installed.")
            return OpenAIProvider()._fallback_analysis(text, analysis_type)
        except Exception as e:
            LOG.error(f"Error with local LLM: {e}")
            return OpenAIProvider()._fallback_analysis(text, analysis_type)
    
    def _analyze_with_server(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """Use a local server API to analyze with a local model"""
        try:
            import requests
            
            # Define server route
            url = f"{self.llm_server}/v1/completions"
            
            # Define prompts for different analysis types (same as local model)
            prompts = {
                "sentiment": f"Analyze the sentiment of the following text and respond with only one word: positive, negative, or neutral.\n\nText: {text}\n\nSentiment:",
                "keywords": f"Extract 5 important keywords from the following text and list them.\n\nText: {text}\n\nKeywords:",
                "summary": f"Summarize the following text in one short paragraph.\n\nText: {text}\n\nSummary:",
                "custom": f"""Analyze the following text and provide marketing recommendations.

Text: {text}

Provide recommendations in this structure:
Tone: 
Keywords: 
Color Scheme: 
Focus: """
            }
            
            prompt = prompts.get(analysis_type, prompts["custom"])
            
            # Call the API
            LOG.info(f"Calling LLM server at {self.llm_server}")
            
            response = requests.post(url, json={
                "prompt": prompt,
                "max_tokens": 256,
                "temperature": 0.1
            }, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("choices", [{}])[0].get("text", "").strip()
                
                # Process result the same way as with local model
                if analysis_type == "sentiment":
                    result_lower = result.lower()
                    if "positive" in result_lower:
                        return {"sentiment": "positive"}
                    elif "negative" in result_lower:
                        return {"sentiment": "negative"}
                    else:
                        return {"sentiment": "neutral"}
                        
                elif analysis_type == "keywords":
                    # Extract keywords from the generated text
                    # Split by commas, newlines, or spaces
                    keywords = []
                    for word in result.replace(',', ' ').replace('\n', ' ').split():
                        word = word.strip('.,!?()[]{};:"\'')
                        if word and len(word) > 2 and word.lower() not in ["and", "the", "for", "keywords", "keyword"]:
                            keywords.append(word)
                    
                    return {"keywords": keywords[:5]}  # Limit to 5 keywords
                    
                elif analysis_type == "summary":
                    return {"summary": result}
                    
                elif analysis_type == "custom":
                    # Try to extract structured information
                    lines = result.split('\n')
                    rec = {}
                    
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip().lower()
                            value = value.strip()
                            
                            if key == "tone":
                                rec["tone"] = value
                            elif key == "keywords":
                                # Split keywords
                                keywords = [k.strip() for k in value.split(',')]
                                rec["keywords"] = [k for k in keywords if k][:5]
                            elif key == "color scheme":
                                rec["color_scheme"] = value
                            elif key == "focus":
                                rec["focus"] = value
                    
                    # Ensure we have all required fields
                    if not all(k in rec for k in ["tone", "keywords", "color_scheme", "focus"]):
                        # Fall back to default values for missing fields
                        default = {
                            "tone": "professional and straightforward",
                            "keywords": ["quality", "value", "reliability", "performance", "solution"],
                            "color_scheme": "neutral and balanced",
                            "focus": "Messaging should emphasize quality and reliability"
                        }
                        
                        for k, v in default.items():
                            if k not in rec or not rec[k]:
                                rec[k] = v
                    
                    return {"result": rec}
                
                else:
                    return {"error": "Unknown analysis type"}
            else:
                LOG.error(f"Server returned error: {response.status_code} - {response.text}")
                return OpenAIProvider()._fallback_analysis(text, analysis_type)
                
        except ImportError:
            LOG.error("Could not import requests. Make sure it's installed.")
            return OpenAIProvider()._fallback_analysis(text, analysis_type)
        except Exception as e:
            LOG.error(f"Error with LLM server: {e}")
            return OpenAIProvider()._fallback_analysis(text, analysis_type)