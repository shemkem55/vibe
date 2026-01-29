"""
Audio Listener Module
Handles microphone input and speech-to-text transcription.
Supports Google Speech Recognition (online) and anticipates local Whisper support.
"""

import os
import time
import os
import time
import threading

try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False
    sr = None

class AudioListener:
    def __init__(self):
        self.recognizer = None
        self.microphone = None
        self.available = False
        
        if not _SR_AVAILABLE:
            print("⚠️ Audio Input Unavailable: 'SpeechRecognition' library not installed.")
            return

        self.recognizer = sr.Recognizer()
        # 🧠 PATIENT LISTENING CONFIGURATION
        self.recognizer.energy_threshold = 300  # Sensitivity (lower = more sensitive)
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 2.5   # Wait 2.5s of silence before stopping (Thinking time)
        self.recognizer.non_speaking_duration = 0.5
        
        self.microphone = None
        self.available = False
        
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.available = True
            print("🎤 Microphone initialized successfully.")
        except ImportError:
            # PyAudio missing
            print("⚠️ Audio Input Unavailable: PyAudio library not found.")
            print("   To enable voice, install system dependencies:")
            print("   Ubuntu/Debian: sudo apt-get install portaudio19-dev && pip install pyaudio")
            print("   Mac: brew install portaudio && pip install pyaudio")
            self.available = False
        except Exception as e:
            # Other device error
            print(f"⚠️ Microphone Not Detected: {e}")
            self.available = False
        except Exception as e:
            # Other device error
            print(f"⚠️ Microphone Not Detected: {e}")
            self.available = False

    def listen(self, timeout=10, phrase_time_limit=45):
        """
        Listen to the microphone and return text.
        timeout: Seconds to wait for speech to start (default 10s)
        phrase_time_limit: Max duration of speech (default 45s)
        """
        if not self.available:
            return None
        
        print("\n🎤 Listening... (Take your time, I'm listening)")
        
        try:
            with self.microphone as source:
                # Short adjustment for ambient noise each time
                # self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            print("⏳ Transcribing...")
            
            # Use Google Web Speech API (default)
            # In production, you might swap this for OpenAI Whisper local
            text = self.recognizer.recognize_google(audio)
            print(f"🗣️  You said: '{text}'")
            return text
            
        except sr.WaitTimeoutError:
            print("❌ Listening timed out (no speech detected).")
            return None
        except sr.UnknownValueError:
            print("❌ Could not understand audio.")
            return None
        except sr.RequestError as e:
            print(f"❌ Could not request results; {e}")
            return None
        except Exception as e:
            print(f"❌ Audio Error: {e}")
            return None

    def is_available(self):
        return self.available

# Standalone test
if __name__ == "__main__":
    listener = AudioListener()
    if listener.is_available():
        listener.listen()
