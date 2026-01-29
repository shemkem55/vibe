import json
import random
import re

class CadenceController:
    def __init__(self, essence_path="essence.json"):
        self.essence = self._load_essence(essence_path)
        self.traits = self.essence.get("speech_traits", {})
        
        # Load specific traits
        self.pause_patterns = self.traits.get("pause_patterns", ["...", "–"])
        self.use_contractions = self.traits.get("contractions", True)
        self.rhythm_type = self.traits.get("rhythm", "flowing")

    def _load_essence(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def apply_cadence(self, text, emotion="neutral"):
        """
        Main pipeline to transform raw text into 'Vibe' speech.
        """
        # 1. Base Cleanup
        processed = text.strip()
        
        # 2. Emotional Resonance Styling
        processed = self._apply_emotional_flavor(processed, emotion)

        # 3. Contractions (if enabled)
        if self.use_contractions:
            processed = self._apply_contractions(processed)
            
        # 4. Rhythmic Pauses (The "Thoughtful" Layer)
        processed = self._inject_pauses(processed)
        
        # 5. Fillers (The "Hesitation" Layer)
        filler_prob = 0.3 if emotion in ["contemplative", "curiosity"] else 0.1
        processed = self._add_fillers(processed, probability=filler_prob)
        
        # 6. Stylistic Formatting
        if self.rhythm_type == "syncopated":
            processed = self._syncopate(processed)
            
        return processed

    def _apply_emotional_flavor(self, text, emotion):
        """Adjust punctuation and intensity based on emotion."""
        if emotion == "joy":
            # Add energy
            text = text.replace(".", "!")
            if not text.endswith("!"): text += "!"
        elif emotion == "sadness":
            # Dim the intensity, more trailing off
            text = text.replace("!", ".")
            if random.random() < 0.5:
                text = text.rstrip(".") + "..."
        elif emotion == "anger":
            # Short, punchy, maybe caps for emphasis (sparingly)
            text = text.upper() if random.random() < 0.1 else text
        elif emotion == "contemplative":
            # More reflective pauses
            if not text.endswith("..."): text += " –"
            
        return text

    def _apply_contractions(self, text):
        replacements = {
            r"\b(I|i) am\b": "I'm",
            r"\b(d)o not\b": "don't",
            r"\b(c)an not\b": "can't",
            r"\b(w)ill not\b": "won't",
            r"\b(i)t is\b": "it's",
            r"\b(t)hat is\b": "that's",
            r"\b(w)hat is\b": "what's",
            r"\b(i)s not\b": "isn't",
            r"\b(a)re not\b": "aren't"
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _inject_pauses(self, text):
        """Insert pauses at logical breakpoints to simulate thinking."""
        # Split into clauses roughly
        words = text.split()
        if len(words) < 5: 
            return text
            
        new_words = []
        for i, word in enumerate(words):
            new_words.append(word)
            
            # Chance to pause after conjunctions or commas
            if word.endswith(',') or word.lower() in ["but", "and", "so", "because"]:
                if random.random() < 0.25:  # 25% chance
                    pause = random.choice(self.pause_patterns)
                    # Don't double up punctuation
                    if word.endswith(','):
                        new_words[-1] = word[:-1] # Remove comma for the pause char
                    new_words.append(pause)
        
        return " ".join(new_words)

    def _add_fillers(self, text, probability=0.1):
        """Add conversational fillers at the start or mid-sentence."""
        fillers = ["you know", "I mean", "sort of", "like", "actually"]
        
        # Start of sentence filler
        if random.random() < probability:
            text = f"{random.choice(fillers)}, {text}"
            
        return text

    def _syncopate(self, text):
        """
        'Syncopated' rhythm: Broken flow, more dashes, lowercase vibes.
        """
        # Lowercase randomly or fully? Let's go for specific style.
        # For 'Vibe' agent, maybe lowercase start if it's casual
        if random.random() < 0.3:
            text = text[0].lower() + text[1:]
            
        return text

if __name__ == "__main__":
    cadence = CadenceController()
    
    raw_responses = [
        "I am not sure about that, but it is an interesting theory.",
        "Cognitive science suggests that memory is reconstructive, not reproductive.",
        "I feel like we are getting somewhere with this project."
    ]
    
    print(f"--- Cadence Test (Rhythm: {cadence.rhythm_type}) ---\n")
    for raw in raw_responses:
        print(f"Raw:   {raw}")
        print(f"Vibe:  {cadence.apply_cadence(raw, emotion='contemplative')}")
        print("-" * 40)
