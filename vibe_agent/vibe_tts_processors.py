# vibe_tts_processors.py
import numpy as np

class SpeechPreprocessor:
    """Convert written text to speakable text with cues"""
    
    @staticmethod
    def add_speech_cues(text, vibe_profile):
        """
        Transform written text for more natural speech.
        """
        # Handle pauses
        if "..." in text:
            pause_time = vibe_profile["pauses"]["ellipsis"]
            # Piper doesn't support [pause:x] directly in string easily without SSML
            # But we can replace with phonetic indicators or multiple dots
            text = text.replace("...", "... ...")
            
        # Convert written contractions to spoken form if not already
        contractions = {
            "cannot": "can not",
            "you're": "you are",
            "don't": "do not",
            "won't": "will not"
        }
        for written, spoken in contractions.items():
            text = text.replace(written, spoken)
        
        return text

class WhisperProcessor:
    """Create whisper/breathy voice effects"""
    
    def apply_whisper_effect(self, audio_array, strength=0.3):
        """
        Apply whisper effect to audio array.
        """
        if strength <= 0:
            return audio_array
        
        # 1. Reduce volume
        audio_array = audio_array * (1 - strength * 0.3)
        
        # 2. Add breath noise
        length = len(audio_array)
        noise = np.random.randn(length) * 0.01 * strength
        
        # Simple envelope to make noise breathe with the signal
        envelope = np.abs(audio_array)
        audio_array = audio_array * 0.8 + (noise * envelope * 2)
        
        return audio_array

class EmotionModulator:
    """Adjust speech based on emotional content"""
    
    def detect_emotion(self, text):
        emotional_words = {
            "joy": ["happy", "excited", "wonderful", "love", "amazing"],
            "sadness": ["sad", "miss", "lonely", "hurt", "heavy"],
            "anger": ["angry", "frustrated", "hate", "annoyed"],
            "surprise": ["wow", "unexpected", "shocked"]
        }
        
        text_lower = text.lower()
        for emotion, words in emotional_words.items():
            if any(word in text_lower for word in words):
                return emotion
        
        return "neutral"
    
    def modulate_params(self, base_params, emotion):
        """Adjust parameters like speed or pitch based on detected emotion."""
        if emotion == "joy":
            base_params['speed'] *= 1.1
            base_params['pitch_variance'] += 0.05
        elif emotion == "sadness":
            base_params['speed'] *= 0.9
            base_params['pitch_variance'] -= 0.05
            
        return base_params
