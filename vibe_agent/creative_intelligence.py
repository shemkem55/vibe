"""
Creative Intelligence Module - AI Upgrade Phase
Handles original content generation, style emulation, and cross-domain synthesis.
"""

import random
from typing import List, Dict, Any

class CreativeIntelligence:
    """Enhanced module for creative and divergent thinking"""
    
    def __init__(self, db_path='agent_memory.db'):
        self.db_path = db_path
        self.styles = {
            "minimalist": "Brief, impactful, using whitespace and precise language.",
            "lovecraftian": "Dense, archaic, atmospheric, focusing on the unknown and vast.",
            "cyberpunk": "Gritty, technical, neon-soaked, focusing on high-tech and low-life.",
            "academic": "Formal, structured, citing hypothetical sources, analytical.",
            "poetic": "Lyrical, rhythmic, using metaphor and imagery."
        }
    
    def emulate_style(self, content: str, style_name: str) -> str:
        """Emulate a specific writing style (to be used with LLM prompts)"""
        style_desc = self.styles.get(style_name, "natural")
        return f"Transform the following content into a {style_name} style. Style description: {style_desc}\n\nContent: {content}"

    def synthesize_insight(self, concept_a: str, concept_b: str) -> Dict[str, Any]:
        """Force a connection between two unrelated concepts (Divergent Thinking)"""
        # Logic for creating a "bridge" between concepts
        connection_types = [
            "Functional Analogy: How A works like B",
            "Structural Similarity: How A's parts mirror B",
            "Relational Shift: What happens if A's rules applied to B",
            "Evolutionary Path: How A could evolve into B"
        ]
        
        connection = random.choice(connection_types)
        
        return {
            "concepts": [concept_a, concept_b],
            "connection_type": connection,
            "seed_insight": f"What if we viewed the {concept_a} through the lens of {concept_b}?"
        }

    def evolve_idea(self, seed_idea: str, iterations: int = 2) -> List[str]:
        """Take an idea through multiple stages of evolution"""
        evolution_stages = [seed_idea]
        
        # In a real implementation, this would involve multiple LLM calls
        # to "Critique", "Expand", and "Synthesize"
        
        return evolution_stages

    def generate_novel_question(self, topic: str) -> str:
        """Generate a thought-provoking, non-obvious question"""
        templates = [
            "If {topic} was a physical landscape, what would its most dangerous feature be?",
            "How would a sentient machine from the year 3000 explain {topic} to a child?",
            "What is the one thing {topic} is trying to hide from us?",
            "If {topic} vanished tomorrow, what would be the first subtle sign of its absence?"
        ]
        
        template = random.choice(templates)
        return template.format(topic=topic)

class StyleEmulationEngine:
    """Specialized engine for mimicking specific voices"""
    
    def __init__(self, llm=None):
        self.llm = llm
    
    def generate_with_style(self, prompt: str, style: str) -> str:
        if not self.llm:
            return f"[Style: {style}] {prompt}"
            
        system_prompt = f"You are an expert at emulating the {style} style. Adjust your tone, vocabulary, and structure accordingly."
        return self.llm.chat([{"role": "user", "content": prompt}], system_prompt=system_prompt)
