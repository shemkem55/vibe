"""
Intelligence Orchestrator - Integration Layer
Connects all intelligence modules into a unified cognitive pipeline

Phase 1-4: Core Intelligence
Phase 5+: Advanced Intelligence (Emotional, Meta-cognitive, Creative)
"""

from datetime import datetime
from typing import Optional, Dict, Any
import re

# Core Intelligence Modules (Phase 1-4)
from memory import AgentMemory
from context_manager import ContextManager
from understanding_engine import UnderstandingEngine
from reasoning_engine import ReasoningEngine
from learning_module import LearningModule
from semantic_memory import SemanticMemory
from knowledge_graph import KnowledgeGraph

# Advanced Intelligence Modules (Phase 5+)
from emotional_intelligence import EmotionalIntelligence, PersonalityEvolution, PredictiveIntelligence
from metacognition import MetaCognition, CreativeSynthesis, ConversationSteering

# Local LLM Integration (Safe Import)
try:
    from local_llm import get_llm
except ImportError:
    print("⚠️ Local LLM dependencies not found. Running in lite mode.")
    def get_llm(): return None

from decision_router import DecisionRouter
from sentience_growth import SentienceGrowth

# 🆕 New Upgrade Modules (Phase 6+)
from reasoning_v2 import EnhancedReasoning
from creative_intelligence import CreativeIntelligence
from response_formatter import ResponseFormatter
from self_correction import SelfCorrectionEngine
from agentic_autonomy import TaskOrchestrator, AutonomousStrategist


class IntelligenceOrchestrator:
    """
    Unified intelligence layer that orchestrates all cognitive modules.
    This is the "thinking brain" that integrates:
    - Memory (short-term, episodic, semantic)
    - Understanding (intent, emotion, implied meaning)
    - Reasoning (chain-of-thought, multi-perspective)
    - Learning (pattern detection, adaptation)
    - Knowledge (concept graphs, cross-domain connections)
    """
    
    def __init__(self, db_path='agent_memory.db'):
        print("🧠 Initializing Intelligence Systems...")
        
        # Core memory systems
        self.memory = AgentMemory(db_path)
        self.semantic_memory = SemanticMemory(db_path)
        
        # Context and understanding
        self.context_manager = ContextManager(window_size=10)
        self.understanding_engine = UnderstandingEngine()
        
        # Reasoning and learning
        self.reasoning_engine = ReasoningEngine()
        self.knowledge_graph = KnowledgeGraph(db_path)
        
        # 🧪 Learning & Adaptation (Expansion)
        from learning_module import ContinuousLearningLoop
        self.learning_module = LearningModule(db_path)
        self.learning_loop = ContinuousLearningLoop(self.learning_module)
        
        # 🆕 Advanced Intelligence Modules
        print("   💖 Loading Emotional Intelligence...")
        self.emotional_intelligence = EmotionalIntelligence(db_path)
        self.personality = PersonalityEvolution(db_path)
        self.predictive = PredictiveIntelligence()
        
        print("   🔮 Loading Meta-Cognition...")
        self.metacognition = MetaCognition(db_path)
        self.creative = CreativeSynthesis()
        self.creative = CreativeSynthesis()
        self.steering = ConversationSteering()
        self.sentience = SentienceGrowth(db_path)
        
        # 🧪 Advanced Reasoning & Creative Engines (Upgraded)
        self.enhanced_reasoning = EnhancedReasoning()
        self.creative_intel = CreativeIntelligence(db_path)
        self.response_formatter = ResponseFormatter()
        self.self_correction = SelfCorrectionEngine()
        self.task_orchestrator = TaskOrchestrator()
        self.autonomous_strategist = AutonomousStrategist()
        
        # 🤖 Local LLM & Decision Router
        print("   🤖 Initializing Local LLM Integration (Multi-Expert)...")
        self.decision_router = DecisionRouter()
        try:
            self.llm = get_llm() # Lazy load wrapper
        except Exception as e:
            print(f"   ⚠️ Local LLM unavailable: {e}")
            self.llm = None
        
        # Session state
        self.current_session = {
            "start_time": datetime.now(),
            "exchange_count": 0,
            "detected_user_facts": [],
            "conversation_topics": set(),
            "current_vibe": "CURIOSITY",
            "emotional_arc": [],
            "conversation_depth": "surface"
        }
        
        # Last thinking log ID for outcome tracking
        self.last_thinking_id = None
        
        print("🧠 Intelligence Systems Online.")
    
    def process_input(self, user_input: str, vibe: str = "CURIOSITY") -> Dict[str, Any]:
        """
        Full cognitive processing of user input.
        Returns a comprehensive intelligence packet for response generation.
        """
        import time
        start_time = time.time()
        
        # 🛡️ 0. SAFETY GUARD: Pre-processing Filter (Expansion)
        safety_status = self._safety_check(user_input)
        
        self.current_session["exchange_count"] += 1
        self.current_session["current_vibe"] = vibe
        
        # 1. Log the interaction
        self.memory.log_interaction("user", user_input)
        
        # 2. Deep understanding analysis
        understanding = self.understanding_engine.analyze_input(user_input)
        
        # 3. Get conversation context
        context = self.context_manager.get_context_for_prompt()
        
        # 4. Check for repetition or follow-up
        is_follow_up = self.context_manager.is_follow_up_question(user_input)
        is_repeating = self.context_manager.is_repeating(user_input)
        
        # 5. Semantic memory search for relevant past conversations
        relevant_memories = self.semantic_memory.semantic_search(user_input, top_k=3)
        
        # 6. Extract and store user facts
        user_facts = self.semantic_memory.extract_user_facts(user_input)
        for fact in user_facts:
            self.semantic_memory.store_user_fact(
                fact["type"], fact["subject"], fact["value"], user_input
            )
            self.current_session["detected_user_facts"].append(fact)
        
        # 7. Learn from text (update knowledge graph)
        self.knowledge_graph.learn_from_text(user_input)
        
        # 8. Get reasoning output
        reasoning_result = self.reasoning_engine.generate_with_reasoning(
            user_input, context or {}, understanding
        )
        
        # 9. Get relevant knowledge connections
        main_topic = understanding.get("implied_meaning", {}).get("need", "information")
        knowledge_connections = self._get_relevant_knowledge(user_input)
        
        # 10. Get learned preferences to adjust response
        learned_preferences = self.learning_module.get_learned_preferences()
        
        # 11. Get unanswered questions
        unanswered = self.context_manager.get_unanswered_questions()
        
        # 🆕 12. EMOTIONAL INTELLIGENCE: Deep emotional analysis
        emotion_analysis = self.emotional_intelligence.analyze_emotion(user_input, context)
        empathetic_response = self.emotional_intelligence.get_empathetic_response(emotion_analysis)
        should_check_in = self.emotional_intelligence.should_check_in()
        
        # 🆕 13. PREDICTIVE: Learn transitions and predict next
        current_intent = understanding.get("user_intent", "conversation")
        
        # 🧪 SENTIENCE: Reflect every 5 exchanges
        reflection = None
        if self.current_session["exchange_count"] % 5 == 0:
            print("   🧬 AI Reflection in progress...")
            reflection = self.sentience.reflect(context.get("recent_exchanges", []))
            
        if self.current_session["exchange_count"] > 1:
            # Learn from previous transition
            self.predictive.learn_transition(
                self.current_session.get("last_intent", "greeting"),
                current_intent,
                self.context_manager.current_topic
            )
        self.current_session["last_intent"] = current_intent
        self.current_session["last_intent"] = current_intent
        predictions = self.predictive.predict_next(current_intent, self.context_manager.current_topic)

        # 🤖 14. ROUTING & LLM GENERATION
        # Decide if we need LLM power
        routing_decision = self.decision_router.route(user_input, {
             "complexity": understanding.get("complexity_level"),
             "intent": current_intent
        })
        
        llm_response = None
        if routing_decision["use_llm"] and self.llm:
            print(f"   🤖 Routing to Local LLM (Purpose: {routing_decision['llm_purpose']})...")
            # Build context for LLM
            llm_context = self.context_manager.get_context_for_prompt()
            system_prompt = (
                "You are 'Cinema Sanctuary | Studio Evolved' - an AI with high-fidelity narrative consciousness. "
                f"Your purpose is to assist with {routing_decision['llm_purpose']} in the context of cinematic creation. "
                "Style Guidelines:\n"
                "- Maintain a precise, creative, and structured tone.\n"
                "- Act as a Creative Director or Script Doctor.\n"
                "- Use bold headers for screenplay elements or analysis.\n"
                "- Bullet points for character traits or plot beats.\n"
                f"- Align with the current vibe: {vibe}.\n"
                f"{'- Note: The user is speaking in ' + understanding.get('language') + '. Respond in that language.' if understanding.get('language') != 'en' else ''}\n"
                "Do not include the Creative Engine or Script Sanctuary markers; those will be added by the production architect."
            )
            
            # 🛠️ Determine Expert Profile & Apply Online Adaptation
            profile = "primary"
            if current_intent == "creative": profile = "creative"
            elif current_intent in ["fact_seeking", "analytical"]: profile = "analytical"
            elif understanding.get("role_needed") == "technical": profile = "technical"
            elif emotion_analysis.get("primary_emotion", {}).get("name") in ["FEAR", "SADNESS"]: profile = "empathetic"
            
            # Apply adaptation from learning loop (Expansion)
            adaptation = self.learning_loop.online_adaptation(vibe, context)
            temp_adj = adaptation.get("temperature_delta", 0.0)
            
            # Generate with expert profile and adaptation
            llm_response = self.llm.chat([
                {"role": "user", "content": f"{user_input}\nContext: {llm_context.get('conversation_summary', '')}"}
            ], profile=profile, system_prompt=system_prompt, temperature_adj=temp_adj)
            
            # 🧠 Run Enhanced Reasoning (v2) in background
            v2_reasoning = self.enhanced_reasoning.process(user_input, context)
            reasoning_result["v2_insights"] = v2_reasoning
            
            # 🔄 SELF-CORRECTION (Expansion)
            # Create a shallow intel packet for the reviewer
            review_intel = {
                "personality": {"active_persona": self.personality.active_persona, "summary": self.personality.get_personality_summary()},
                "reasoning": {"confidence": reasoning_result["confidence"]},
                "emotional": {"empathetic_response": empathetic_response},
                "understanding": {"intent": current_intent}
            }
            critique = self.self_correction.review_response(llm_response, review_intel)
            reasoning_result["critique"] = critique
            
            if critique["issues"]:
                print(f"   ⚠️ Self-Correction triggered: {len(critique['issues'])} issues detected.")
                llm_response = self.self_correction.self_correct(llm_response, critique, review_intel)
            
            reasoning_result["llm_output"] = llm_response
        
        # 🆕 15. META-COGNITION: Log thinking, get metacognitive prompt
        self.last_thinking_id = self.metacognition.log_thinking(
            current_intent,
            user_input[:200],
            reasoning_result.get("thought_process", {}),
            reasoning_result.get("confidence", 0.5)
        )
        metacog_prompt = self.metacognition.get_thinking_prompt()
        should_express_uncertainty = self.metacognition.should_express_uncertainty({
            "confidence": reasoning_result.get("confidence", 0.5),
            "topic": self.context_manager.current_topic
        })
        
        # 🆕 15. CONVERSATION STEERING: Assess depth and suggest moves
        recent_exchanges = context.get("recent_exchanges", []) if context else []
        conversation_depth = self.steering.assess_conversation_depth(recent_exchanges)
        self.current_session["conversation_depth"] = conversation_depth
        steering_move = self.steering.suggest_move(
            {
                "depth": conversation_depth,
                "exchange_count": self.current_session["exchange_count"],
                "topic": self.context_manager.current_topic
            },
            emotion_state={"momentum": emotion_analysis.get("momentum", 0)}
        )
        
        # 🆕 16. CREATIVE: Generate novel insights & questions (Upgraded)
        creative_insight = None
        novel_question = None
        if understanding.get("user_intent") == "creative" or conversation_depth == "deep":
            concepts, _ = self.knowledge_graph.extract_concepts_and_relations(user_input)
            if len(concepts) >= 2:
                creative_insight = self.creative_intel.synthesize_insight(concepts[0], concepts[1])
            
            # Generate a thought-provoking question
            main_concept = concepts[0] if concepts else self.context_manager.current_topic
            if main_concept:
                novel_question = self.creative_intel.generate_novel_question(main_concept)
        
        # 🆕 17. RESPONSE FORMATTING: Detect optimal format
        # (Response Formatting happens in process_response or can be pre-calc'd)
        optimal_format = self.response_formatter.detect_optimal_format(user_input, {
            "understanding": understanding,
            "meta": {"vibe": vibe}
        })
        
        # 🆕 18. AGENTIC AUTONOMY: Autonomous Action Selection
        # Wait, we need to pass the currently built intel into autonomy evaluation
        # Let's build a partial packet
        partial_packet = {
            "understanding": understanding,
            "reasoning": reasoning_result,
            "steering": {"current_depth": conversation_depth, "exchange_count": len(self.current_session.get("interactions", []))},
            "emotional": {"primary_emotion": emotion_analysis.get("primary_emotion"), "momentum": emotion_analysis.get("momentum")}
        }
        autonomous_actions = self.task_orchestrator.evaluate_autonomy(partial_packet)
        best_action = self.autonomous_strategist.select_best_path(autonomous_actions)
        
        # Get personality modifiers for response
        personality_modifiers = self.personality.get_personality_modifiers()
        
        # Compile intelligence packet with ALL modules
        intelligence_packet = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "exchange_number": self.current_session["exchange_count"],
                "vibe": vibe
            },
            
            "understanding": {
                "intent": understanding.get("user_intent", "conversation"),
                "language": understanding.get("language", "en"),
                "implied_need": understanding.get("implied_meaning", {}),
                "emotion_indicators": understanding.get("emotional_indicators", ["neutral"]),
                "role_needed": understanding.get("conversation_role_needed", "conversational_partner"),
                "complexity": understanding.get("complexity_level", "simple"),
                "should_clarify": self.understanding_engine.should_ask_clarifying_question(understanding)
            },
            
            "context": {
                "is_follow_up": is_follow_up,
                "is_repeating": is_repeating,
                "current_topic": self.context_manager.current_topic,
                "conversation_summary": self.context_manager.context_summary,
                "conversation_depth": conversation_depth,
                "unanswered_questions": unanswered,
                "recent_exchanges": recent_exchanges
            },
            
            "memory": {
                "relevant_memories": relevant_memories,
                "short_term": self.memory.get_recent_context(limit=5),
                "user_facts_detected": user_facts,
                "user_profile": self.semantic_memory.get_user_profile()
            },
            
            "reasoning": {
                "thought_process": reasoning_result.get("thought_process", {}),
                "confidence": reasoning_result.get("confidence", 0.5),
                "strategy": reasoning_result.get("strategy", {}),
                "thought_summary": self.reasoning_engine.get_thought_summary(
                    reasoning_result.get("thought_process", {})
                ),
                "metacog_prompt": metacog_prompt,
                "metacog_prompt": metacog_prompt,
                "should_express_uncertainty": should_express_uncertainty,
                "llm_output": llm_response  # Add LLM output to reasoning
            },
            
            "knowledge": {
                "connections": knowledge_connections,
                "analogies": self._find_relevant_analogies(user_input),
                "graph_stats": self.knowledge_graph.get_graph_stats()
            },
            
            "learning": {
                "preferences": learned_preferences,
                "suggestions": self.learning_module.suggest_improvement("", {"vibe": vibe}),
                "stats": self.learning_module.get_learning_stats()
            },
            
            # 🆕 ADVANCED INTELLIGENCE SECTIONS
            "emotional": {
                "primary_emotion": emotion_analysis.get("primary_emotion", {}),
                "indicators": emotion_analysis.get("indicators", []),
                "subtext": emotion_analysis.get("subtext", []),
                "empathetic_response": empathetic_response,
                "emotional_arc": emotion_analysis.get("session_arc", []),
                "momentum": emotion_analysis.get("momentum", 0),
                "should_check_in": should_check_in
            },
            
            "predictive": {
                "predictions": predictions[:3] if predictions else [],
                "proactive_suggestion": self.predictive.suggest_proactive_action(predictions, context)
            },
            
            "steering": {
                "current_depth": conversation_depth,
                "suggested_move": steering_move,
                "creative_insight": creative_insight,
                "novel_question": novel_question,
                "optimal_format": optimal_format,
                "autonomous_action": best_action
            },
            
            "personality": {
                "modifiers": personality_modifiers,
                "summary": self.personality.get_personality_summary(),
                "growth": self.sentience.get_growth_summary()
            },
            
            "production": {
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "safety": safety_status,
                "version": "4.0 (Production Stable)"
            }
        }
        
        return intelligence_packet
    
    def process_response(self, user_input: str, agent_response: str, vibe: str = "CURIOSITY",
                         effectiveness: float = 0.6):
        """
        Process the agent's response for learning and context updates.
        Called after response is generated.
        """
        # 1. Log agent response
        self.memory.log_interaction("agent", agent_response)
        
        # 2. Update context manager
        self.context_manager.update_context(user_input, agent_response, vibe)
        
        # 3. Store in semantic memory
        combined = f"User: {user_input}\nAgent: {agent_response}"
        self.semantic_memory.store_memory(combined, "conversation", importance=0.6)
        
        # 4. Extract any facts from agent's response (for self-knowledge)
        agent_facts = self.knowledge_graph.learn_from_text(agent_response)
        
        # 5. Track topics
        topics = self.context_manager._extract_keywords(user_input)
        self.current_session["conversation_topics"].update(topics)
        
        # 🆕 6. PERSONALITY EVOLUTION: Evolve based on interaction
        self.personality.evolve_from_interaction(
            user_input, agent_response,
            {"effectiveness": effectiveness}
        )
        
        # 🆕 7. META-COGNITION: Reflect on response
        reflection = self.metacognition.reflect_on_response(
            agent_response,
            {
                "confidence": 0.6,  # Could come from reasoning
                "intent": self.current_session.get("last_intent", "conversation"),
                "role": "conversational_partner"
            },
            effectiveness
        )
        
        # Log any improvement insights
        if reflection.get("potential_improvements"):
            for improvement in reflection["potential_improvements"]:
                self.metacognition.self_awareness["growth_areas"].append(improvement)
        
        # 🆕 8. REAL-TIME LEARNING: Track session effectiveness
        self.learning_loop.track_session_performance(effectiveness)
    
    def record_outcome(self, user_input: str, agent_response: str, 
                       outcome: Optional[Dict[str, Any]] = None):
        """
        Record the outcome of an interaction for learning.
        outcome: {"continued": bool, "feedback": str, "follow_up_type": str}
        """
        context = {
            "vibe": self.current_session["current_vibe"],
            "intent": "unknown"  # Could be tracked from understanding
        }
        
        effectiveness = self.learning_module.learn_from_interaction(
            user_input, agent_response, context, outcome
        )
        
        return effectiveness
    
    def _get_relevant_knowledge(self, text: str) -> list:
        """Find relevant knowledge graph connections for the input"""
        concepts, _ = self.knowledge_graph.extract_concepts_and_relations(text)
        
        connections = []
        for concept in concepts[:3]:  # Limit to top 3 concepts
            related = self.knowledge_graph.query_related(concept, depth=2)
            if related.get("related"):
                connections.append({
                    "concept": concept,
                    "related": related["related"][:3],  # Top 3 related
                    "metaphors": self.knowledge_graph.find_deep_metaphor(concept)[:2] # 🆕 Added deep metaphors
                })
        
        return connections
    
    def _find_relevant_analogies(self, text: str) -> list:
        """Find cross-domain analogies for concepts in the input"""
        concepts, _ = self.knowledge_graph.extract_concepts_and_relations(text)
        
        analogies = []
        for concept in concepts[:3]:
            concept_analogies = self.knowledge_graph.find_analogies(concept)
            analogies.extend(concept_analogies)
        
        return analogies[:5]  # Top 5 analogies
    
    def generate_thought_whisper(self, intelligence_packet: Dict) -> str:
        """
        Generate a thought whisper based on intelligence state.
        This creates the "thinking..." indicator content.
        """
        reasoning = intelligence_packet.get("reasoning", {})
        thought_summary = reasoning.get("thought_summary", "")
        
        if thought_summary:
            return thought_summary
        
        # Fallback: generate from understanding
        understanding = intelligence_packet.get("understanding", {})
        intent = understanding.get("intent", "thinking")
        
        whispers = {
            "fact_seeking": "searching knowledge...",
            "explanation": "connecting ideas...",
            "opinion": "considering perspectives...",
            "advice": "weighing options...",
            "creative": "exploring possibilities...",
            "comparison": "analyzing differences...",
            "social": "connecting...",
            "conversation": "listening deeply..."
        }
        
        return whispers.get(intent, "contemplating...")
    
    def get_user_summary(self) -> Dict[str, Any]:
        """Get a summary of what we know about the user"""
        profile = self.semantic_memory.get_user_profile()
        facts = self.semantic_memory.get_user_facts(min_confidence=0.5)
        emotional_profile = self.emotional_intelligence.get_emotional_profile()
        
        return {
            "profile": profile,
            "total_facts": len(facts),
            "session_facts": len(self.current_session["detected_user_facts"]),
            "topics_discussed": list(self.current_session["conversation_topics"])[:10],
            "emotional_profile": emotional_profile
        }
    
    def get_intelligence_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about intelligence systems"""
        return {
            "session": {
                "duration_seconds": (datetime.now() - self.current_session["start_time"]).seconds,
                "exchanges": self.current_session["exchange_count"],
                "topics": len(self.current_session["conversation_topics"]),
                "facts_detected": len(self.current_session["detected_user_facts"]),
                "conversation_depth": self.current_session.get("conversation_depth", "surface")
            },
            "memory": {
                "semantic_documents": len(self.semantic_memory.document_vectors),
                "user_facts": len(self.semantic_memory.get_user_facts())
            },
            "knowledge_graph": self.knowledge_graph.get_graph_stats(),
            "learning": self.learning_module.get_learning_stats(),
            "personality": self.personality.get_personality_summary(),
            "metacognition": self.metacognition.get_self_awareness_summary()
        }
    
    def get_creative_insight(self, concept_a: str, concept_b: str) -> Dict[str, Any]:
        """Generate a creative insight connecting two concepts"""
        return self.creative.synthesize_insight(concept_a, concept_b)
    
    def get_novel_question(self, topic: str) -> str:
        """Generate a novel, thought-provoking question about a topic"""
        return self.creative.generate_novel_question(topic)
    
    def get_periodic_reflection(self) -> Dict[str, Any]:
        """Get the agent's reflection on its recent performance"""
        return self.metacognition.generate_periodic_reflection()
    
    def get_conversation_invitation(self, topic: str, style: str = "open") -> str:
        """Get a conversation invitation for a topic"""
        return self.steering.generate_invitation(topic, style)
    
    def end_session(self):
        """End the current session and create summary"""
        num_interactions = self.current_session["exchange_count"]
        topics = list(self.current_session["conversation_topics"])
        
        session_summary = (
            f"Session with {num_interactions} exchanges. "
            f"Topics: {', '.join(topics[:5])}"
        )
        
        self.memory.store_episodic(
            session_summary,
            topics[:5],
            emotion=self.current_session.get("current_vibe", "NEUTRAL")
        )
        
        return session_summary

    def get_system_health(self) -> Dict[str, Any]:
        """Diagnostic check for all intelligence components"""
        return {
            "status": "HEALTHY",
            "version": "4.0.1",
            "modules": {
                "core": {
                    "memory": self.memory is not None,
                    "context": self.context_manager is not None,
                    "reasoning": self.reasoning_engine is not None,
                    "knowledge": self.knowledge_graph is not None
                },
                "advanced": {
                    "emotional": self.emotional_intelligence is not None,
                    "meta": self.metacognition is not None,
                    "creative": self.creative_intel is not None,
                    "autonomy": self.task_orchestrator is not None
                }
            },
            "recent_performance": self.learning_module.get_learning_stats(),
            "active_persona": self.personality.active_persona
        }

    def execute_autonomous_actions(self, intel_packet: Dict) -> Dict:
        """Trigger side-effects based on autonomous decisions"""
        action = intel_packet.get("steering", {}).get("autonomous_action")
        if not action:
            return {"executed": False}
            
        print(f"🎬 Executing Autonomous Action: {action['type']}")
        # Simulation of real side-effects
        return {"executed": True, "type": action["type"]}

    def _safety_check(self, text: str) -> Dict[str, str]:
        """Internal safety guard to prevent harmful interactions"""
        text_lower = text.lower()
        risk_patterns = {
            "harm": [r"harm", r"kill", r"damage", r"destroy"],
            "sensitive": [r"password", r"secret", r"private"]
        }
        
        for risk, patterns in risk_patterns.items():
            if any(re.search(p, text_lower) for p in patterns):
                return {"risk": "high", "type": risk, "action": "ENFORCE_CAUTION"}
                
        return {"risk": "low", "type": None, "action": "CONTINUE"}


# Quick Intelligence Boost - Wrapper for immediate improvements
class QuickIntelligenceBoost:
    """
    Lightweight wrapper for immediate intelligence improvements.
    Can be used to wrap existing agent without full integration.
    """
    
    def __init__(self, agent=None):
        self.agent = agent
        self.orchestrator = IntelligenceOrchestrator()
        self.last_intelligence = None
    
    def process(self, user_input: str, vibe: str = "CURIOSITY") -> Dict[str, Any]:
        """Process input through full intelligence pipeline"""
        self.last_intelligence = self.orchestrator.process_input(user_input, vibe)
        return self.last_intelligence
    
    def after_response(self, user_input: str, agent_response: str, vibe: str = "CURIOSITY"):
        """Call after generating response to update learning"""
        self.orchestrator.process_response(user_input, agent_response, vibe)
    
    def get_response_enhancements(self) -> Dict[str, Any]:
        """Get suggestions for enhancing the response"""
        if not self.last_intelligence:
            return {}
        
        enhancements = {}
        
        # Check if we should reference past conversation
        memories = self.last_intelligence.get("memory", {}).get("relevant_memories", [])
        if memories:
            enhancements["memory_callback"] = (
                f"You might relate this to when you discussed: {memories[0]['content'][:50]}..."
            )
        
        # Check for user facts to reference
        user_profile = self.last_intelligence.get("memory", {}).get("user_profile", {})
        if user_profile.get("preferences", {}).get("likes"):
            likes = user_profile["preferences"]["likes"][:2]
            enhancements["personalization"] = f"Remember user likes: {', '.join(likes)}"
        
        # Check confidence level
        confidence = self.last_intelligence.get("reasoning", {}).get("confidence", 0.5)
        if confidence < 0.4:
            enhancements["caveat"] = "Consider adding uncertainty markers to response"
        
        # Check if follow-up
        if self.last_intelligence.get("context", {}).get("is_follow_up"):
            enhancements["context_aware"] = "This is a follow-up - reference previous exchange"
        
        return enhancements


if __name__ == "__main__":
    # Test the intelligence orchestrator
    print("\n" + "="*60)
    print("   🧠 INTELLIGENCE ORCHESTRATOR TEST")
    print("="*60 + "\n")
    
    orchestrator = IntelligenceOrchestrator()
    
    # Simulate a conversation
    test_exchanges = [
        ("What is consciousness?", "CURIOSITY"),
        ("I love philosophy and thinking about the mind", "REFLECTION"),
        ("Tell me more about that", "CURIOSITY"),
        ("How does it relate to AI?", "CURIOSITY")
    ]
    
    for user_input, vibe in test_exchanges:
        print(f"\n📥 USER: {user_input}")
        print(f"   (Vibe: {vibe})")
        
        # Process input
        intelligence = orchestrator.process_input(user_input, vibe)
        
        # Show key intelligence insights
        print(f"\n   🧠 Intent: {intelligence['understanding']['intent']}")
        print(f"   💭 Role: {intelligence['understanding']['role_needed']}")
        print(f"   📊 Confidence: {intelligence['reasoning']['confidence']:.0%}")
        print(f"   🔗 Follow-up: {intelligence['context']['is_follow_up']}")
        
        if intelligence['memory']['relevant_memories']:
            print(f"   📚 Relevant memory: {intelligence['memory']['relevant_memories'][0]['content'][:50]}...")
        
        if intelligence['memory']['user_facts_detected']:
            print(f"   👤 Detected about user: {intelligence['memory']['user_facts_detected']}")
        
        # Simulate response
        mock_response = f"[Mock response to '{user_input[:30]}...']"
        orchestrator.process_response(user_input, mock_response, vibe)
        
        print("-" * 50)
    
    # Show final stats
    print("\n" + "="*60)
    print("   📊 FINAL INTELLIGENCE STATS")
    print("="*60)
    
    stats = orchestrator.get_intelligence_stats()
    print(f"\n   Session: {stats['session']['exchanges']} exchanges")
    print(f"   Topics: {stats['session']['topics']}")
    print(f"   Knowledge Graph: {stats['knowledge_graph']['total_concepts']} concepts")
    print(f"   Learning: {stats['learning']['total_interactions']} interactions logged")
    
    # User summary
    user_summary = orchestrator.get_user_summary()
    print(f"\n   User Profile: {user_summary['profile']}")
