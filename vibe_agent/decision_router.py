"""
Decision Router
Decides whether to route queries to Rule-Based systems or the LLM.
"""

class DecisionRouter:
    """Decide when to use LLM vs rule-based systems"""
    
    LLM_TRIGGERS = {
        "complex_reasoning": [
            "why", "explain", "analyze", "compare", "contrast",
            "what if", "how would", "imagine", "suppose",
            "meaning of", "significance of"
        ],
        "creative_tasks": [
            "write a", "create a", "compose", "imagine",
            "story", "poem", "analogy", "metaphor"
        ],
        "deep_understanding": [
            "what does this mean", "philosophical",
            "ethical", "moral"
        ],
        "problem_solving": [
            "how to solve", "how can i", "what should i do",
            "recommend", "troubleshoot"
        ]
    }
    
    RULE_BASED_TRIGGERS = {
        "factual_questions": [
            "what is", "who is", "when did", "where is",
            "how many", "which", "capital of", "population of"
        ],
        "simple_responses": [
            "hello", "hi", "thanks", "thank you", "bye",
            "good morning", "how are you"
        ]
    }
    
    def route(self, user_input, analysis):
        """Decide which systems to use"""
        input_lower = user_input.lower()
        
        decision = {
            "use_llm": False,
            "llm_purpose": None,
            "use_rules": True,
            "needs_research": False,
            "confidence_threshold": 0.7
        }
        
        # Check for LLM triggers
        for purpose, triggers in self.LLM_TRIGGERS.items():
            if any(trigger in input_lower for trigger in triggers):
                decision["use_llm"] = True
                decision["llm_purpose"] = purpose
                break
        
        # Check for factual questions (need research)
        if any(trigger in input_lower for trigger in self.RULE_BASED_TRIGGERS["factual_questions"]):
            decision["needs_research"] = False # Let research engine handle this logic separately
        
        # Override: If very short or simple, use rules only
        if len(user_input.split()) < 3 and "why" not in input_lower:
            decision["use_llm"] = False
            decision["use_rules"] = True
        
        # Override: If conversation is deep, use LLM
        # (This relies on analysis passed in)
        if analysis.get("complexity") == "complex":
            decision["use_llm"] = True
            
        return decision
