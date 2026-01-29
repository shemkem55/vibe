import json
import re
from datetime import datetime
from memory import AgentMemory

class InputProcessor:
    def __init__(self, essence_path="essence.json"):
        self.memory = AgentMemory()
        self.essence = self._load_essence(essence_path)
        
        # Simple keyword maps for the "Phase 4.1" non-API approach
        self.emotion_map = {
            "joy": ["happy", "excited", "love", "great", "awesome", "yes", "yay"],
            "sadness": ["sad", "blue", "down", "sorry", "bad", "miss"],
            "curiosity": ["why", "how", "what", "tell", "explain", "wonder"],
            "anger": ["hate", "mad", "angry", "stop", "no"],
            "contemplative": ["think", "feel", "believe", "maybe", "perhaps"]
        }

    def _load_essence(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"name": "Vibe", "speech_traits": {}, "knowledge_mood": {}}

    def detect_emotion(self, text):
        """
        Rudimentary emotional analysis based on keyword density.
        Returns: primary_emotion, confidence_score
        """
        text_lower = text.lower()
        scores = {k: 0 for k in self.emotion_map}
        
        words = re.findall(r'\w+', text_lower)
        total_words = len(words) if words else 1
        
        for word in words:
            for emotion, keywords in self.emotion_map.items():
                if word in keywords:
                    scores[emotion] += 1
        
        # Find max score
        best_emotion = max(scores, key=scores.get)
        if scores[best_emotion] == 0:
            return "neutral", 0.0
        
        confidence = scores[best_emotion] / total_words
        return best_emotion, round(confidence, 2)

    def classify_intent(self, text):
        """Determine if user is asking, stating, greeting, or ending."""
        text_lower = text.lower()
        
        # Helper for word boundary matching
        def has_word(words, txt):
            pattern = r'\b(' + '|'.join(map(re.escape, words)) + r')\b'
            return bool(re.search(pattern, txt))

        if has_word(["bye", "goodbye", "later", "see ya"], text_lower):
            return "farewell"
        
        if has_word(["hi", "hello", "hey", "greetings", "yo"], text_lower):
            return "greeting"
            
        if "?" in text or any(text_lower.startswith(w) for w in ["who", "what", "where", "when", "why", "how", "can"]):
            return "query"
            
        return "statement"

    def process(self, user_text):
        """
        Main pipeline: Input -> Analysis -> Memory -> Context
        """
        timestamp = datetime.now().isoformat()
        
        # 1. Analyze
        emotion, confidence = self.detect_emotion(user_text)
        intent = self.classify_intent(user_text)
        
        # 2. Memory Operations
        # A. Log this interaction
        self.memory.log_interaction("user", user_text)
        
        # B. Recall relevant past episodic memories (simple keyword search)
        keywords = [w for w in user_text.split() if len(w) > 4] # heuristic for "important" words
        relevant_memories = self.memory.recall(keywords) if keywords else []
        
        # C. Get recent session context
        recent_history = self.memory.get_recent_context(limit=5)

        # 3. Assemble Context Object
        context_packet = {
            "meta": {
                "timestamp": timestamp,
                "agent_name": self.essence.get("name", "Agent"),
            },
            "user_analysis": {
                "text": user_text,
                "detected_emotion": emotion,
                "emotion_confidence": confidence,
                "intent": intent
            },
            "memory_context": {
                "short_term": recent_history,
                "long_term_hits": relevant_memories
            }
        }
        
        return context_packet

if __name__ == "__main__":
    # Test the pipeline
    processor = InputProcessor()
    
    test_inputs = [
        "Hello there!",
        "I am feeling a bit sad today because I lost my keys.",
        "What do you think about the stars?",
        "This is amazing, I love it!"
    ]
    
    print(f"--- Processing Pipeline Test for agent: {processor.essence.get('name')} ---\n")
    
    for txt in test_inputs:
        print(f"User: '{txt}'")
        ctx = processor.process(txt)
        print(f" > Emotion: {ctx['user_analysis']['detected_emotion']} ({ctx['user_analysis']['emotion_confidence']})")
        print(f" > Intent:  {ctx['user_analysis']['intent']}")
        print(f" > Memory Hits: {len(ctx['memory_context']['long_term_hits'])}")
        print("-" * 40)
