import re

class QuestionDetector:
    """Determine if input is a direct question needing factual answer."""
    
    QUESTION_PATTERNS = {
        "factual": [
            r"^what (is|are|was|were) (the |a |an )?.*\?$",
            r"^how (many|much|far|long|old|tall|deep) .*\?$",
            r"^when (did|was|will|does) .*\?$",
            r"^where (is|are|was|were) .*\?$",
            r"^who (is|are|was|were|invented|created|wrote|discovered) .*\?$",
            r"^which .*\?$",
        ],
        "procedural": [
            r"^how (to|do|can|should) (i |you |we )?.*\?$",
            r"^what (should|would|could) (i |you |we ).*\?$",
        ],
        "definition": [
            r"^(define|explain|describe) .*",
            r"^what does .* mean\?$",
        ]
    }
    
    def classify_question(self, user_input):
        """Categorize the type of question."""
        input_lower = user_input.lower().strip()
        
        # Check for question mark
        if "?" not in input_lower:
            return {"is_question": False, "type": None, "needs_direct_answer": False}
        
        # Check patterns
        for q_type, patterns in self.QUESTION_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, input_lower):
                    return {
                        "is_question": True,
                        "type": q_type,
                        "needs_direct_answer": True
                    }
        
        # Default: it's a question but maybe conversational
        return {
            "is_question": True,
            "type": "conversational",
            "needs_direct_answer": False
        }
    
    def extract_core_query(self, user_input):
        """Strip away conversational fluff to get the core question."""
        prefixes_to_remove = [
            "hey, ", "hi, ", "hello, ", "so, ", "well, ", "actually, ",
            "i was wondering ", "can you tell me ", "do you know ",
            "could you please ", "would you mind telling me "
        ]
        
        core = user_input.lower()
        for prefix in prefixes_to_remove:
            if core.startswith(prefix):
                core = core[len(prefix):]
        
        core = core.rstrip("? ").strip()
        return core

class DirectAnswerFormatter:
    """Extract and format direct answers from research results."""
    
    def extract_direct_answer(self, research_result, question_type):
        """Get the most direct answer from research summary."""
        if not research_result or not research_result.get("summary"):
            return None
            
        summary = research_result["summary"]
        
        # For factual questions, extract first sentence
        if question_type in ["factual", "definition"]:
            sentences = summary.split(". ")
            if sentences:
                return self._clean_answer(sentences[0])
        
        # For procedural, look for steps
        elif question_type == "procedural":
            return self._extract_steps(summary)
        
        # Default: return first 150 chars
        if len(summary) > 150:
            return summary[:147] + "..."
        return summary
    
    def _clean_answer(self, answer):
        """Remove verbose intros and keep core answer."""
        # Remove common verbose starters
        removals = [
            "According to ", "Based on ", "It appears that ",
            "The research shows that ", "Studies indicate that "
        ]
        
        for removal in removals:
            if answer.startswith(removal):
                answer = answer[len(removal):]
        
        # Capitalize first letter
        if answer:
            answer = answer[0].upper() + answer[1:]
            
        return answer.strip()
    
    def _extract_steps(self, text):
        """Extract numbered steps if present."""
        lines = text.split("\n")
        steps = [line for line in lines if re.match(r"^\d+\.", line.strip())]
        
        if steps:
            return "\n".join(steps)
        return text
    
    def format_with_citation(self, answer, citations):
        """Add minimal citation to direct answer."""
        if not citations:
            return answer
            
        source = citations[0]
        source_name = source.get("source", "")
        
        citation_map = {
            "wikipedia": "(Wikipedia)",
            "arxiv": "(academic research)",
            "news": "(recent news)",
            "duckduckgo": ""
        }
        
        citation = citation_map.get(source_name, "")
        if citation:
            return f"{answer} {citation}"
        return answer
