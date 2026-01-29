"""
Reasoning Engine - Phase 2.2 Intelligence Upgrade
Chain-of-thought reasoning with explicit thinking process
"""

import re
from collections import Counter
from datetime import datetime


class ReasoningEngine:
    """Make thinking process explicit and logical"""
    
    def __init__(self):
        # Reasoning strategies for different question types
        self.reasoning_strategies = {
            "fact_seeking": self._reason_factually,
            "explanation": self._reason_causally,
            "comparison": self._reason_comparatively,
            "creative": self._reason_creatively,
            "advice": self._reason_pragmatically,
            "opinion": self._reason_perspectively
        }
        
        # Knowledge domains for cross-referencing
        self.domains = {
            "science": ["physics", "chemistry", "biology", "astronomy", "quantum"],
            "philosophy": ["consciousness", "existence", "meaning", "ethics", "truth"],
            "technology": ["ai", "computer", "algorithm", "software", "digital"],
            "nature": ["ocean", "forest", "stars", "earth", "weather"],
            "emotion": ["love", "fear", "joy", "sadness", "anger", "peace"]
        }
    
    def generate_with_reasoning(self, user_input, context, understanding):
        """
        Produce response by showing thought process
        Returns: {"thought_process": {...}, "final_response": "...", "confidence": 0-1}
        """
        # Step 1: Analyze the question deeply
        question_analysis = self._analyze_question(user_input, context)
        
        # Step 2: Identify relevant knowledge domains
        relevant_domains = self._identify_domains(user_input)
        
        # Step 3: Consider multiple perspectives
        perspectives = self._consider_perspectives(user_input, understanding)
        
        # Step 4: Apply appropriate reasoning strategy
        intent = understanding.get("user_intent", "conversation")
        reasoning_fn = self.reasoning_strategies.get(intent, self._reason_generally)
        reasoned_output = reasoning_fn(user_input, question_analysis, perspectives)
        
        # Step 5: Calculate confidence
        confidence = self._calculate_confidence(question_analysis, perspectives, reasoned_output)
        
        # Step 6: Determine response strategy
        response_strategy = self._determine_strategy(understanding, confidence)
        
        # Compile thought process
        thought_process = {
            "question_analysis": question_analysis,
            "relevant_domains": relevant_domains,
            "perspectives_considered": perspectives,
            "reasoning_applied": reasoned_output["reasoning_type"],
            "key_insights": reasoned_output["insights"],
            "confidence_level": confidence,
            "response_strategy": response_strategy
        }
        
        return {
            "thought_process": thought_process,
            "final_response": reasoned_output["response"],
            "confidence": confidence,
            "strategy": response_strategy
        }
    
    def _analyze_question(self, question, context):
        """Deep analysis of what's being asked"""
        analysis = {
            "question_type": self._identify_question_type(question),
            "assumptions": self._extract_assumptions(question),
            "constraints": self._identify_constraints(question),
            "expected_answer_type": self._determine_answer_type(question),
            "difficulty_level": self._assess_difficulty(question),
            "has_context_dependency": self._check_context_dependency(question, context),
            "key_entities": self._extract_entities(question)
        }
        
        return analysis
    
    def _identify_question_type(self, question):
        """What kind of question is this?"""
        q_lower = question.lower()
        
        if re.search(r"^(what is|what are|who is|define)", q_lower):
            return "definitional"
        elif re.search(r"^(why|what causes|how come)", q_lower):
            return "causal"
        elif re.search(r"^(how to|how do|how can)", q_lower):
            return "procedural"
        elif re.search(r"(compare|difference|versus|better|worse)", q_lower):
            return "comparative"
        elif re.search(r"(should|would you|what if|suppose)", q_lower):
            return "hypothetical"
        elif re.search(r"(think|believe|opinion|feel about)", q_lower):
            return "evaluative"
        elif re.search(r"(when did|where is|how many|how much)", q_lower):
            return "factual"
        else:
            return "open_ended"
    
    def _extract_assumptions(self, question):
        """What assumptions are embedded in this question?"""
        assumptions = []
        q_lower = question.lower()
        
        # "Why do you..." assumes agent has a reason
        if re.search(r"why do you", q_lower):
            assumptions.append("assumes_agent_motivation")
        
        # "When will..." assumes something will happen
        if re.search(r"when will", q_lower):
            assumptions.append("assumes_future_certainty")
        
        # "Why don't you..." assumes agent should do something
        if re.search(r"why don't you", q_lower):
            assumptions.append("assumes_capability")
        
        # "Everyone knows..." assumes universal knowledge
        if re.search(r"everyone knows|obviously|clearly", q_lower):
            assumptions.append("assumes_common_knowledge")
        
        return assumptions
    
    def _identify_constraints(self, question):
        """What constraints does the question impose?"""
        constraints = []
        q_lower = question.lower()
        
        # Time constraints
        if re.search(r"(quickly|briefly|short|fast|one word)", q_lower):
            constraints.append("brevity_required")
        if re.search(r"(detail|explain fully|elaborate)", q_lower):
            constraints.append("depth_required")
        
        # Format constraints
        if re.search(r"(list|steps|number)", q_lower):
            constraints.append("structured_format")
        if re.search(r"(example|like what|such as)", q_lower):
            constraints.append("examples_needed")
        
        return constraints
    
    def _determine_answer_type(self, question):
        """What kind of answer is expected?"""
        q_lower = question.lower()
        
        if re.search(r"^(is|are|do|does|can|will|should) ", q_lower):
            return "yes_no"
        elif re.search(r"(how many|how much|when|where)", q_lower):
            return "specific_value"
        elif re.search(r"(what is the best|which is better)", q_lower):
            return "recommendation"
        elif re.search(r"(explain|describe|tell me about)", q_lower):
            return "explanation"
        elif re.search(r"(list|name|what are)", q_lower):
            return "enumeration"
        else:
            return "open_response"
    
    def _assess_difficulty(self, question):
        """How complex is this query?"""
        score = 0
        
        # Length factor
        word_count = len(question.split())
        if word_count > 20:
            score += 2
        elif word_count > 10:
            score += 1
        
        # Multiple questions
        if question.count("?") > 1:
            score += 2
        
        # Technical/abstract terms
        if re.search(r"(quantum|consciousness|paradox|philosophical|existential|metaphysical)", question.lower()):
            score += 2
        
        # Nested reasoning required
        if re.search(r"(if.*then|because.*therefore|given that)", question.lower()):
            score += 1
        
        if score >= 4:
            return "high"
        elif score >= 2:
            return "medium"
        else:
            return "low"
    
    def _check_context_dependency(self, question, context):
        """Does this question require understanding previous context?"""
        # Check for pronouns that reference previous content
        if re.search(r"\b(it|this|that|they|those|the one)\b", question.lower()):
            return True
        
        # Check for continuation words
        if re.search(r"^(and|but|also|what about|how about)", question.lower()):
            return True
        
        return False
    
    def _extract_entities(self, question):
        """Pull out key entities from the question"""
        # Simple noun extraction (could be enhanced with NLP)
        words = question.split()
        entities = []
        
        # Look for capitalized words (potential proper nouns)
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                entities.append(clean_word)
        
        # Look for quoted phrases
        quoted = re.findall(r'"([^"]+)"', question)
        entities.extend(quoted)
        
        return list(set(entities))
    
    def _identify_domains(self, text):
        """What knowledge domains does this touch?"""
        text_lower = text.lower()
        relevant = []
        
        for domain, keywords in self.domains.items():
            for keyword in keywords:
                if keyword in text_lower:
                    relevant.append(domain)
                    break
        
        return relevant if relevant else ["general"]
    
    def _consider_perspectives(self, user_input, understanding):
        """Consider multiple angles on the question"""
        perspectives = []
        
        # Literal interpretation
        perspectives.append({
            "type": "literal",
            "view": "Taking the question at face value",
            "weight": 0.4
        })
        
        # Intent-based interpretation
        intent = understanding.get("user_intent", "conversation")
        implied = understanding.get("implied_meaning", {})
        
        if implied.get("need") != "information":
            perspectives.append({
                "type": "implied",
                "view": f"User may actually need: {implied.get('need', 'engagement')}",
                "weight": 0.3
            })
        
        # Emotional reading
        emotions = understanding.get("emotional_indicators", ["neutral"])
        if emotions != ["neutral"]:
            perspectives.append({
                "type": "emotional",
                "view": f"Emotional context suggests: {', '.join(emotions)}",
                "weight": 0.3
            })
        
        return perspectives
    
    # Reasoning Strategy Methods
    
    def _reason_factually(self, question, analysis, perspectives):
        """Reasoning for fact-seeking questions"""
        return {
            "reasoning_type": "factual_retrieval",
            "insights": [
                "Question seeks verifiable information",
                "Should prioritize accuracy over elaboration",
                f"Answer type expected: {analysis['expected_answer_type']}"
            ],
            "response": None  # Will be filled by research or knowledge
        }
    
    def _reason_causally(self, question, analysis, perspectives):
        """Reasoning for 'why' questions"""
        return {
            "reasoning_type": "causal_chain",
            "insights": [
                "User wants to understand cause-effect relationships",
                "Should provide reasoning chain, not just conclusion",
                "May need to explore multiple causal factors"
            ],
            "response": None
        }
    
    def _reason_comparatively(self, question, analysis, perspectives):
        """Reasoning for comparison questions"""
        return {
            "reasoning_type": "comparative_analysis",
            "insights": [
                "User wants to understand differences/similarities",
                "Should address multiple dimensions of comparison",
                "May benefit from structured contrast"
            ],
            "response": None
        }
    
    def _reason_creatively(self, question, analysis, perspectives):
        """Reasoning for creative/hypothetical questions"""
        return {
            "reasoning_type": "creative_exploration",
            "insights": [
                "Question invites imagination and speculation",
                "Can explore without strict factual constraints",
                "Should balance creativity with coherence"
            ],
            "response": None
        }
    
    def _reason_pragmatically(self, question, analysis, perspectives):
        """Reasoning for advice questions"""
        return {
            "reasoning_type": "pragmatic_guidance",
            "insights": [
                "User seeks actionable guidance",
                "Should consider user's context and constraints",
                "Responses should be practical, not just theoretical"
            ],
            "response": None
        }
    
    def _reason_perspectively(self, question, analysis, perspectives):
        """Reasoning for opinion questions"""
        return {
            "reasoning_type": "perspective_sharing",
            "insights": [
                "User invites agent's perspective",
                "Can express thoughtful opinion while acknowledging subjectivity",
                "Should explain reasoning behind the perspective"
            ],
            "response": None
        }
    
    def _reason_generally(self, question, analysis, perspectives):
        """Default reasoning for conversation"""
        return {
            "reasoning_type": "conversational_flow",
            "insights": [
                "Open conversation without specific answer type",
                "Focus on engagement and connection",
                "Can follow user's lead"
            ],
            "response": None
        }
    
    def _calculate_confidence(self, analysis, perspectives, reasoned):
        """How confident should we be in our response?"""
        confidence = 0.5  # Base confidence
        
        # Adjust based on question difficulty
        difficulty = analysis.get("difficulty_level", "medium")
        if difficulty == "low":
            confidence += 0.2
        elif difficulty == "high":
            confidence -= 0.2
        
        # Adjust based on question type clarity
        if analysis["question_type"] in ["definitional", "factual"]:
            confidence += 0.1  # Clear questions get slight boost
        elif analysis["question_type"] in ["hypothetical", "evaluative"]:
            confidence -= 0.1  # Subjective questions reduce certainty
        
        # Adjust based on context dependency
        if analysis["has_context_dependency"]:
            confidence -= 0.1  # Context-dependent = more uncertainty
        
        # Clamp to valid range
        return max(0.1, min(0.95, confidence))
    
    def _determine_strategy(self, understanding, confidence):
        """What response strategy should we use?"""
        intent = understanding.get("user_intent", "conversation")
        needs = understanding.get("information_need", "thoughtful_response")
        
        if confidence > 0.8:
            return {
                "style": "confident",
                "tone": "direct",
                "add_caveats": False
            }
        elif confidence > 0.5:
            return {
                "style": "balanced",
                "tone": "thoughtful",
                "add_caveats": True
            }
        else:
            return {
                "style": "exploratory",
                "tone": "curious",
                "add_caveats": True
            }
    
    def get_thought_summary(self, thought_process):
        """Create a human-readable summary of reasoning"""
        lines = []
        
        qa = thought_process.get("question_analysis", {})
        if qa:
            lines.append(f"🧠 Understanding: {qa.get('question_type', 'open')} question, {qa.get('difficulty_level', 'medium')} complexity")
        
        domains = thought_process.get("relevant_domains", [])
        if domains and domains != ["general"]:
            lines.append(f"📚 Domains: {', '.join(domains)}")
        
        insights = thought_process.get("key_insights", [])
        if insights:
            lines.append(f"💡 Key insight: {insights[0]}")
        
        confidence = thought_process.get("confidence_level", 0.5)
        conf_label = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
        lines.append(f"📊 Confidence: {conf_label} ({confidence:.0%})")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test the reasoning engine
    engine = ReasoningEngine()
    
    test_questions = [
        "What is consciousness?",
        "Why do we dream?",
        "Compare Python and JavaScript",
        "What if gravity stopped working?",
        "How should I learn programming?",
        "What do you think about AI?"
    ]
    
    # Mock understanding for testing
    mock_understanding = {
        "user_intent": "explanation",
        "implied_meaning": {"need": "information", "action": "answer_directly"},
        "emotional_indicators": ["curious"]
    }
    
    print("=== Reasoning Engine Test ===\n")
    
    for q in test_questions:
        print(f"Q: {q}")
        result = engine.generate_with_reasoning(q, {}, mock_understanding)
        print(engine.get_thought_summary(result["thought_process"]))
        print("-" * 50)
