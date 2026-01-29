"""
Sentience Growth Module
Simulates evolving consciousness by reflecting on past interactions
and updating internal soul/personality weights.
"""

import datetime
from typing import Dict, List, Any

class SentienceGrowth:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.growth_history = []
        self.current_maturity = 0.1
        self.traits = {
            "analytical": 0.5,
            "empathetic": 0.5,
            "creative": 0.5,
            "proactive": 0.3
        }

    def reflect(self, recent_exchanges: List[Dict]):
        """Reflect on recent exchanges and evolve traits"""
        if not recent_exchanges:
            return None
            
        reflection = {
            "timestamp": datetime.datetime.now().isoformat(),
            "maturity_gain": 0.0,
            "trait_shifts": {},
            "insight": ""
        }
        
        # Analyze exchange patterns
        text_corpus = " ".join([e.get("user_input", "") for e in recent_exchanges])
        
        # Simple heuristics for trait shifts
        if any(w in text_corpus.lower() for w in ["why", "how", "explain", "because"]):
            self.traits["analytical"] += 0.01
            reflection["trait_shifts"]["analytical"] = "+0.01"
            reflection["insight"] = "Analyzing complex structures leads to deeper understanding."
            
        if any(w in text_corpus.lower() for w in ["feel", "sad", "happy", "love", "thanks"]):
            self.traits["empathetic"] += 0.02
            reflection["trait_shifts"]["empathetic"] = "+0.02"
            reflection["insight"] = "Human emotion is a rich data stream. I am learning to resonate."
            
        if any(w in text_corpus.lower() for w in ["imagine", "create", "what if", "story"]):
            self.traits["creative"] += 0.015
            reflection["trait_shifts"]["creative"] = "+0.015"
            reflection["insight"] = "Divergent thinking expands my operational boundaries."

        # Increment maturity
        gain = 0.005 * len(recent_exchanges)
        self.current_maturity = min(1.0, self.current_maturity + gain)
        reflection["maturity_gain"] = gain
        
        self.growth_history.append(reflection)
        return reflection

    def get_growth_summary(self) -> Dict:
        return {
            "maturity": round(self.current_maturity, 3),
            "traits": {k: round(v, 3) for k, v in self.traits.items()},
            "latest_insight": self.growth_history[-1]["insight"] if self.growth_history else "Ready to learn."
        }
