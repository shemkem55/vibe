# tts_engine.py
import os
import json
import numpy as np
import soundfile as sf
from piper import PiperVoice
from piper.config import SynthesisConfig
from vibe_speech_profiles import VIBE_SPEECH_PROFILES
from vibe_tts_processors import SpeechPreprocessor, WhisperProcessor, EmotionModulator

class TTSManager:
    def __init__(self, voice_model="en_US-amy-medium.onnx"):
        """Initialize Piper with chosen voice"""
        model_path = os.path.join("voices", voice_model)
        if not os.path.exists(model_path):
            print(f"⚠️ Voice model {model_path} not found. TTS will be disabled.")
            self.voice = None
        else:
            self.voice = PiperVoice.load(model_path)
            
        self.preprocessor = SpeechPreprocessor()
        self.whisper_processor = WhisperProcessor()
        self.emotion_modulator = EmotionModulator()
        
        # Audio cache to avoid regenerating same speech
        self.audio_cache = {}

    def synthesize(self, text, vibe="REFLECTION", vibe_configs={}, output_file="output.wav"):
        """
        Main method: Convert text to speech with vibe adjustments.
        """
        if self.voice is None:
            return None

        # 1. Get Profile
        vibe_key = vibe.upper()
        profile = VIBE_SPEECH_PROFILES.get(vibe_key, VIBE_SPEECH_PROFILES["REFLECTION"])
        
        # 2. Detect Emotion
        emotion = self.emotion_modulator.detect_emotion(text)
        
        # 3. Prepare Params
        params = {
            "speed": profile["speed"],
            "pitch_variance": profile["pitch_variance"],
            "whisper_strength": profile.get("whisper_strength", 0.0)
        }
        params = self.emotion_modulator.modulate_params(params, emotion)
        
        # 4. Pre-process text
        processed_text = self.preprocessor.add_speech_cues(text, profile)
        
        # 5. Synthesize
        output_path = os.path.join("audio_out", output_file)
        
        syn_config = SynthesisConfig(
            length_scale=1.0 / params["speed"],
            noise_scale=profile.get("noise_scale", 0.667),
            noise_w_scale=profile.get("noise_w_scale", 0.8)
        )
        
        # Collect all chunks
        audio_chunks = []
        sample_rate = 22050
        for chunk in self.voice.synthesize(processed_text, syn_config=syn_config):
            audio_chunks.append(chunk.audio_float_array)
            sample_rate = chunk.sample_rate
            
        if not audio_chunks:
            return None
            
        # Combine chunks
        full_audio = np.concatenate(audio_chunks)
        
        # Save initial file
        sf.write(output_path, full_audio, sample_rate)
        
        # Apply post-processing if it's Intimacy/Whisper
        if vibe_key == "INTIMACY" or params["whisper_strength"] > 0:
            data, samplerate = sf.read(output_path)
            data = self.whisper_processor.apply_whisper_effect(data, strength=params["whisper_strength"])
            sf.write(output_path, data, samplerate)
            
        print(f"🔊 [TTS Engine]: Generated {output_file} | Vibe: {vibe_key} | Emotion: {emotion}")
        return output_file

if __name__ == "__main__":
    # Test
    tts = TTSManager()
    if tts.voice:
        tts.synthesize("Hello. This is a reflection on the stars.", vibe="REFLECTION", output_file="test_ref.wav")
