import random
import re
from collections import defaultdict

class ThoughtStream:
    def __init__(self, depth=2):
        self.chain = defaultdict(list)
        self.depth = depth
        self.used_phrases = []
        self.corpus = [
            "the stars are just holes in the ceiling of the world.",
            "patterns of light dancing on the water, like thoughts in a dream.",
            "resonance and rhythm make the universe move.",
            "cognitive science is just the study of ghosts in the machine.",
            "urban weeds growing through concrete are signs of a silent rebellion.",
            "the space between the notes is where the music really lives.",
            "everything is a fractal of something else.",
            "memories of forgotten conversations...",
            "tracing patterns in the silence...",
            "echoes of what wasn't said...",
            "the space between words...",
            "a thought unfolding like origami..."
        ]
        self._build_chain()

    def _build_chain(self):
        for text in self.corpus:
            words = text.split()
            if len(words) <= self.depth:
                continue
            for i in range(len(words) - self.depth):
                key = tuple(words[i:i + self.depth])
                next_word = words[i + self.depth]
                self.chain[key].append(next_word)

    def learn_from_text(self, text):
        """Append new text to the chain to evolve the thought stream."""
        words = text.split()
        if len(words) <= self.depth:
            return
        for i in range(len(words) - self.depth):
            key = tuple(words[i:i + self.depth])
            next_word = words[i + self.depth]
            self.chain[key].append(next_word)

    def generate_seed(self, length=8):
        """Generate a random 'thought fragment'."""
        if not self.chain:
            return "..."
            
        try:
            start_key = random.choice(list(self.chain.keys()))
        except IndexError:
            return "tracing patterns..."
            
        result = list(start_key)
        
        for _ in range(length - self.depth):
            key = tuple(result[-self.depth:])
            if key in self.chain:
                result.append(random.choice(self.chain[key]))
            else:
                break
                
        return " ".join(result)

    def generate_dream_fragment(self, user_input=""):
        """Create unique neural drift each time."""
        # 1. Extract potential themes from input
        words = [w.lower() for w in user_input.split() if len(w) > 4]
        theme = random.choice(words) if words else random.choice(["silence", "echoes", "drifting", "memory"])
        
        patterns = [
            "echoes of {theme}...",
            "tracing {theme} in the quiet...",
            "memories of {theme} unfolding...",
            "the space where {theme} lives...",
            "a drift of {theme} through the mind...",
            "{theme} acting as a mirror...",
            "resonance between {theme} and stars..."
        ]
        
        # 2. Generate and check for uniqueness
        attempts = 0
        while attempts < 5:
            drift = random.choice(patterns).format(theme=theme)
            if drift not in self.used_phrases:
                break
            attempts += 1
            
        self.used_phrases.append(drift)
        if len(self.used_phrases) > 20:
            self.used_phrases.pop(0) # Keep history short
            
        return drift

if __name__ == "__main__":
    ts = ThoughtStream()
    print("--- 🧠 Thought Stream Simulator (Phase 1.2) ---")
    print("Generated Seeds:")
    for _ in range(5):
        print(f" > {ts.generate_seed()}")
