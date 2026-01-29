# vibe_speech_profiles.py

VIBE_SPEECH_PROFILES = {
    "REFLECTION": {
        "speed": 1.15,  # Slightly faster
        "pitch_variance": 0.15,  # More melodic
        "pauses": {
            "comma": 0.3,  # seconds
            "period": 0.5,
            "ellipsis": 0.8
        },
        "emphasis_words": ["what", "how", "why", "curious", "wonder"],
        "breath_pattern": "frequent_light",
        "warmth": 0.8,
        "prosody": "rising",
        "ambient_sound": "soft_wind_chimes"
    },
    
    "REFLECTION": {
        "speed": 0.85,  # Slower
        "pitch_variance": 0.05,  # More monotone, thoughtful
        "pauses": {
            "comma": 0.5,
            "period": 0.9,
            "ellipsis": 1.2
        },
        "emphasis_words": ["think", "feel", "remember", "perhaps", "reflect"],
        "breath_pattern": "deep_spaced",
        "warmth": 0.6,
        "prosody": "falling",
        "ambient_sound": "distant_piano"
    },
    
    "INTIMACY": {
        "speed": 0.95,
        "pitch_variance": 0.08,
        "pauses": {
            "comma": 0.4,
            "period": 0.6,
            "ellipsis": 1.0
        },
        "emphasis_words": ["you", "i", "we", "together", "here", "quiet"],
        "breath_pattern": "soft_close",
        "warmth": 0.9,
        "prosody": "soft_rising",
        "whisper_strength": 0.3, 
        "ambient_sound": "room_tone"
    },
    
    "PLAYFUL": {
        "speed": 1.25,
        "pitch_variance": 0.25,  # Very melodic
        "pauses": {
            "comma": 0.2,
            "period": 0.4,
            "ellipsis": 0.6
        },
        "emphasis_words": ["fun", "imagine", "what if", "maybe", "energy", "wild"],
        "breath_pattern": "quick_light",
        "warmth": 0.7,
        "prosody": "bouncy",
        "ambient_sound": "light_bells"
    }
}
