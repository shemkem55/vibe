"""
Understanding Engine - Phase 2 Intelligence Upgrade
Deep analysis of user intent, implied meaning, and emotional subtext
"""

import re

class UnderstandingEngine:
    """Go beyond keyword matching to true comprehension"""
    
    def analyze_input(self, text, context=None):
        """Deep analysis of user input"""
        language = self._detect_language(text)
        return {
            "literal_meaning": text,
            "language": language,
            "implied_meaning": self._extract_implied(text),
            "user_intent": self._classify_intent(text),
            "emotional_indicators": self._detect_emotion_indicators(text),
            "information_need": self._identify_needs(text),
            "conversation_role_needed": self._determine_role(text),
            "complexity_level": self._assess_complexity(text),
            "key_concepts": self._extract_key_concepts(text)
        }
    
    def _extract_implied(self, text):
        """What is the user REALLY asking/saying?"""
        text_lower = text.lower()
        
        # Patterns that reveal deeper meaning
        implied_patterns = {
            # Confusion signals
            r"(i don't understand|confused|lost|unclear)": {
                "need": "simplification",
                "action": "explain_simpler"
            },
            
            # Depth request
            r"(tell me more|elaborate|go deeper|expand on)": {
                "need": "depth",
                "action": "expand_details"
            },
            
            # Causality quest
            r"^why": {
                "need": "causality",
                "action": "explain_reasons"
            },
            
            # Process understanding
            r"^how (do|does|can|to)": {
                "need": "process",
                "action": "describe_steps"
            },
            
            # Verification
            r"(really|seriously|are you sure|is that true)": {
                "need": "verification",
                "action": "provide_evidence"
            },
            
            # Interest signal
            r"(interesting|fascinating|cool|wow)": {
                "need": "expansion",
                "action": "suggest_related"
            },
            
            # Comparison
            r"(compare|difference between|versus|vs|rather than)": {
                "need": "comparison",
                "action": "contrast_items"
            }
        }
        
        for pattern, meaning in implied_patterns.items():
            if re.search(pattern, text_lower):
                return meaning
        
        return {"need": "information", "action": "answer_directly"}
    
    def _classify_intent(self, text):
        """What does the user want from this interaction?"""
        text_lower = text.lower()
        
        # Intent detection patterns
        if re.search(r"(what is|what are|who invented|when did|how many)", text_lower):
            return "fact_seeking"
        
        if re.search(r"(explain|why does|how does|what causes)", text_lower):
            return "explanation"
        
        if re.search(r"(what do you think|in your opinion|how do you feel)", text_lower):
            return "opinion"
        
        if re.search(r"(what should|how can i|what would you do|advice)", text_lower):
            return "advice"
        
        if re.search(r"(what does .* mean|can you clarify|unclear)", text_lower):
            return "clarification"
        
        if re.search(r"(hello|hi|hey|good morning|how are you)", text_lower):
            return "social"
        
        if re.search(r"(imagine|what if|suppose|hypothetically)", text_lower):
            return "creative"
        
        if re.search(r"(compare|difference|better|worse|versus)", text_lower):
            return "comparison"
        
        return "conversation"
    
    def _detect_emotion_indicators(self, text):
        """Detect emotional markers in text"""
        indicators = []
        text_lower = text.lower()
        
        # Positive indicators
        if re.search(r"(love|amazing|wonderful|great|excellent|perfect)", text_lower):
            indicators.append("positive")
        
        # Negative indicators
        if re.search(r"(hate|terrible|awful|bad|worst|horrible)", text_lower):
            indicators.append("negative")
        
        # Confusion
        if re.search(r"(confused|lost|don't understand|unclear)", text_lower):
            indicators.append("confused")
        
        # Curiosity
        if re.search(r"(curious|wonder|interesting|fascinating)", text_lower):
            indicators.append("curious")
        
        # Frustration
        if re.search(r"(frustrated|annoyed|tired of|keep asking)", text_lower):
            indicators.append("frustrated")
        
        # Excitement  
        if re.search(r"(!+|wow|omg|amazing|incredible)", text_lower):
            indicators.append("excited")
        
        return indicators if indicators else ["neutral"]
    
    def _identify_needs(self, text):
        """What does the user need from this response?"""
        intent = self._classify_intent(text)
        
        needs_map = {
            "fact_seeking": "accurate_information",
            "explanation": "clear_understanding",
            "opinion": "perspective",
            "advice": "actionable_guidance",
            "clarification": "simpler_explanation",
            "social": "friendly_engagement",
            "creative": "imaginative_exploration",
            "comparison": "analytical_breakdown"
        }
        
        return needs_map.get(intent, "thoughtful_response")
    
    def _determine_role(self, text):
        """What role should the agent play in response?"""
        intent = self._classify_intent(text)
        
        role_mapping = {
            "fact_seeking": "knowledgeable_guide",
            "explanation": "patient_teacher",
            "opinion": "thoughtful_companion",
            "advice": "wise_advisor",
            "clarification": "clear_explainer",
            "social": "friendly_presence",
            "creative": "creative_partner",
            "comparison": "analytical_mind"
        }
        
        return role_mapping.get(intent, "conversational_partner")
    
    def _assess_complexity(self, text):
        """How complex is this query?"""
        # Simple heuristics
        word_count = len(text.split())
        has_multiple_questions = text.count("?") > 1
        has_technical_terms = re.search(r"(quantum|algorithm|neural|molecular|philosophical)", text.lower())
        
        complexity_score = 0
        
        if word_count > 20:
            complexity_score += 1
        if has_multiple_questions:
            complexity_score += 1
        if has_technical_terms:
            complexity_score += 1
        
        if complexity_score >= 2:
            return "complex"
        elif complexity_score == 1:
            return "moderate"
        else:
            return "simple"
    
    def should_ask_clarifying_question(self, analysis):
        """Determine if we need to ask for clarification"""
        # Ask for clarification if:
        # 1. Intent is unclear
        # 2. User seems confused
        # 3. Question is very complex
        
        if analysis["user_intent"] == "conversation" and "?" in analysis["literal_meaning"]:
            return True
        
        if "confused" in analysis["emotional_indicators"]:
            return True
        
        if analysis["complexity_level"] == "complex" and len(analysis["literal_meaning"]) < 50:
            # Complex topic but short question - might need clarification
            return True
        
        return False

    def _extract_key_concepts(self, text):
        """Extract main conceptual entities from text"""
        # Simple heuristic: longer words, capitalized words, excluding common stop words
        stop_words = {"what", "how", "the", "a", "an", "is", "are", "was", "were", "to", "for", "with", "in", "on", "at"}
        words = re.findall(r'\b\w+\b', text)
        
        concepts = []
        for word in words:
            if word.lower() not in stop_words and len(word) > 4:
                concepts.append(word.lower())
        
        # Add capitalized sequences as concepts
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        concepts.extend([p.lower() for p in proper_nouns])
        
        return list(set(concepts))

    def _detect_language(self, text):
        """Detect the primary language of the text (Heuristic)"""
        text_lower = text.lower()
        # Clean punctuation for better matching
        clean_text = re.sub(r'[^\w\s]', '', text_lower)
        
        lang_markers = {
            "en": ["the", "this", "what", "how", "you"],
            "es": ["el", "la", "que", "como", "usted", "hola", "estás"],
            "fr": ["le", "la", "quel", "comment", "vous"],
            "de": ["der", "die", "das", "was", "wie", "sie"],
            "it": ["il", "la", "che", "come", "voi"]
        }
        
        scores = {lang: 0 for lang in lang_markers}
        words = clean_text.split()
        
        for word in words:
            for lang, markers in lang_markers.items():
                if word in markers:
                    scores[lang] += 1
        
        best_lang = max(scores, key=scores.get)
        return best_lang if scores[best_lang] > 0 else "en"
