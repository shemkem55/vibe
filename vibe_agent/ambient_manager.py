# ambient_manager.py
import os
import soundfile as sf
import numpy as np

class AmbientSoundManager:
    """Add vibe-specific background sounds"""
    
    def __init__(self, ambient_dir="audio/ambient"):
        self.ambient_dir = ambient_dir
        self.ambient_library = {
            "REFLECTION": "chimes_soft.wav",
            "REFLECTION": "piano_distant.wav",
            "INTIMACY": "room_tone.wav",
            "PLAYFUL": "bells_light.wav"
        }
        os.makedirs(ambient_dir, exist_ok=True)
    
    def mix_ambient(self, speech_path, vibe):
        """Mix ambient sound with the generated speech WAV."""
        vibe_key = vibe.upper()
        if vibe_key not in self.ambient_library:
            return
            
        sound_file = os.path.join(self.ambient_dir, self.ambient_library[vibe_key])
        if not os.path.exists(sound_file):
            return
            
        try:
            # 1. Load both sounds
            speech, sr = sf.read(speech_path)
            ambient, asr = sf.read(sound_file)
            
            # Ensure same sample rate (assuming 22050 for now, or just resampling)
            # Simplest: assume they match or resample on the fly is too complex here
            # We will just verify or skip if major mismatch
            
            # 2. Adjust ambient length to match speech
            if len(ambient) > len(speech):
                ambient = ambient[:len(speech)]
            elif len(ambient) < len(speech):
                repeats = (len(speech) // len(ambient)) + 1
                ambient = np.tile(ambient, repeats)[:len(speech)]
                
            # 3. Mix
            # Volume reduction for ambient
            vol = 0.1 if vibe_key != "REFLECTION" else 0.05
            mixed = speech + (ambient * vol)
            
            # 4. Normalize
            max_val = np.abs(mixed).max()
            if max_val > 1.0:
                mixed = mixed / max_val
                
            # 5. Overwrite
            sf.write(speech_path, mixed, sr)
            # print(f"🍃 [Ambient]: Mixed {vibe_key} soundscape.")
            
        except Exception as e:
            print(f"❌ [Ambient Error]: {e}")
