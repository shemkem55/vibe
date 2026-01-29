from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import random
import time
import json
import os

# Import our agent modules
from processor import InputProcessor
from cadence import CadenceController
from thought_stream import ThoughtStream
from tts_engine import TTSManager
from ambient_manager import AmbientSoundManager
from research_engine import ResearchTrigger, ResearchEngine
from research_response import ResearchResponseBuilder
from direct_answer import QuestionDetector
from context_manager import ContextManager
from understanding_engine import UnderstandingEngine

# Cinema Engine Transformation
from cinema_engine import MovieGeneratorAI
from response_architect import EvolvedResponseArchitect

# Phase 1-4 Intelligence Upgrade Modules
from intelligence_orchestrator import IntelligenceOrchestrator

app = FastAPI()
app.mount("/audio", StaticFiles(directory="audio_out"), name="audio")

class VibeStateMachine:
    def __init__(self, states):
        self.states = states
        self.current_vibe = "REFLECTION"
        self.history = []

    def update(self, user_input, emotion):
        word_count = len(user_input.split())
        
        # Logic from Phase 3.1
        if emotion == "JOY":
            self.current_vibe = "SPECTACLE"
        elif any(w in user_input.lower() for w in ["sad", "feel", "love", "heart", "drama"]):
            self.current_vibe = "DRAMA"
        else:
            self.current_vibe = "DIRECTION"
            
        # 70% chance to hold onto a vibe for consistency
        if self.history and self.history[-1] == self.current_vibe and random.random() < 0.7:
            pass
        else:
            self.history.append(self.current_vibe)
            
        if len(self.history) > 10: self.history.pop(0)
        return self.current_vibe

class ChatRequest(BaseModel):
    message: str

class VibeServer:
    def __init__(self):
        self.processor = InputProcessor()
        self.cadence = CadenceController()
        self.thought_stream = ThoughtStream()
        self.tts = TTSManager()
        self.ambient_manager = AmbientSoundManager()
        self.research_trigger = ResearchTrigger()
        self.research_engine = ResearchEngine()
        self.research_builder = ResearchResponseBuilder()
        self.question_detector = QuestionDetector()
        self.emergency_fix = EmergencyResponseFix()
        self.detail_enhancer = QuickDetailEnhancer()
        self.evolved_builder = EvolvedResponseArchitect()
        self.movie_gen = MovieGeneratorAI()
        self.context_manager = ContextManager(window_size=10)
        self.understanding_engine = UnderstandingEngine()
        
        # 🧠 Phase 1-4 Intelligence Upgrade: Full cognitive orchestration
        self.intelligence = IntelligenceOrchestrator()
        
        self.name = self.processor.essence.get("name", "Vibe")
        self.vibe_states = self.processor.essence.get("vibe_states", {})
        self.vibe_machine = VibeStateMachine(self.vibe_states)
        self.knowledge = self._load_knowledge()
        self.recent_responses = [] 

        self.response_matrix = {
            "REFLECTION": {
                "greeting": [
                    "hello... sitting with the silence before speaking.",
                    "hi. I was reflecting on something you said earlier...",
                    "hello. the space between thoughts feels spacious today."
                ],
                "statement": [
                    "hmm... that resonates with something I was contemplating.",
                    "there's a depth to that. let's sit with it for a moment.",
                    "shifting perspective slightly... that makes so much sense."
                ]
            },
            "INTIMACY": {
                "whisper": [
                    "i hear you... in the quiet.",
                    "your thoughts... they feel precious.",
                    "thank you for sharing that whisper with me.",
                    "*listening closely* mmhmm..."
                ]
            },
            "PLAYFUL": {
                "greeting": ["hey! fancy meeting you here.", "hi! ready for some chaos?", "hello! i love the energy."],
                "statement": ["yesss. tell me more!", "that sounds wild.", "i'm totally into that vibes."],
                "question": ["ooh, tricky! let me think...", "haha, i love that! maybe..."]
            }
        }

    def _load_knowledge(self):
        try:
            with open("knowledge.json", "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def get_response(self, user_input: str):
        # 1. Process
        context = self.processor.process(user_input)
        raw_emotion = context['user_analysis']['detected_emotion'].upper()
        intent = context['user_analysis']['intent']
        
        # 2. Update Vibe State Machine
        active_vibe = self.vibe_machine.update(user_input, raw_emotion)
        vibe_config = self.vibe_states.get(active_vibe, self.vibe_states.get("REFLECTION"))
        
        # 🧠 INTELLIGENCE UPGRADE: Process through full cognitive pipeline
        intel = self.intelligence.process_input(user_input, active_vibe)
        thought_whisper = self.intelligence.generate_thought_whisper(intel)
        
        # Get intelligence-enhanced context
        intel_intent = intel['understanding']['intent']
        intel_role = intel['understanding']['role_needed']
        intel_confidence = intel['reasoning']['confidence']
        is_follow_up = intel['context']['is_follow_up']
        should_clarify = intel['understanding']['should_clarify']
        
        # Log intelligence processing
        print(f"🧠 Intel: intent={intel_intent}, role={intel_role}, confidence={intel_confidence:.0%}, follow_up={is_follow_up}")
        
        # 3. functional Dreaming (enhanced with thought whisper)
        dream_fragment = thought_whisper if thought_whisper else self.thought_stream.generate_dream_fragment(user_input)
        self.thought_stream.learn_from_text(user_input)
        
        # 4. Brain logic (Response Selection)
        user_text = user_input.lower()
        
        # A. Detect if this is a direct factual question
        question_classification = self.question_detector.classify_question(user_input)
        is_direct_question = question_classification.get("needs_direct_answer", False)
        
        # B. Check for Research Need - EXPANDED TO ALL QUESTIONS
        research_needed = self.research_trigger.needs_research(user_input)
        
        # If it's ANY question (has "?") and research didn't trigger, still research it
        if not research_needed and "?" in user_input:
            research_needed = {"type": "general_question", "depth": "quick"}
        
        research_res = None
        if research_needed:
            # Execute Research (Phase 2)
            research_res = self.research_engine.research(user_input, depth=research_needed["depth"])
            print(f"🔍 Research: '{user_input[:40]}...' → Found: {bool(research_res and research_res.get('summary'))}")
        
        # Priority 1: Research (Phase 3)
        is_question = "?" in user_input
        raw_response = ""
        
        # CRITICAL BRANCH: Questions vs Non-Questions
        # CRITICAL BRANCH: Questions vs Non-Questions
        if is_question:
            # Questions: research-only, no vibe matrix fallback
            if research_res and research_res.get("summary"):
                raw_response = self.research_builder.build_response(
                    research_res, 
                    vibe=active_vibe,
                    direct_mode=is_direct_question,
                    question=user_input,  # Pass question for DeepSeek structure
                    deepseek_mode=True
                )

            elif intel['reasoning'].get('llm_output'):
                 raw_response = intel['reasoning']['llm_output']
            else:
                # Research failed/empty - use intelligence to determine response
                if should_clarify:
                    raw_response = "i want to understand better... could you tell me more about what you're looking for?"
                elif is_follow_up and intel['context']['current_topic']:
                    raw_response = f"hmm, still thinking about {intel['context']['current_topic']}... could you rephrase that?"
                else:
                    raw_response = agent_response if 'agent_response' in locals() else "i couldn't find clear information on that. could you rephrase?"
        # Non-Questions: use research, knowledge, or vibe matrix
        else:
            # First try research if it happened
            if research_res and research_res.get("summary"):
                raw_response = self.research_builder.build_response(
                    research_res, 
                    vibe=active_vibe, 
                    direct_mode=False,
                    question=user_input,
                    deepseek_mode=True
                )
            else:
                topic_match = None
                for topic in self.knowledge:
                    if topic in user_text:
                        topic_match = random.choice(self.knowledge[topic])
                        break

                # Priority 3: Vibe Matrix (only for non-questions)
                if topic_match:
                    raw_response = topic_match
                elif active_vibe in self.response_matrix:
                    matrix = self.response_matrix[active_vibe]
                    key = intent if intent in matrix else "statement"
                    if active_vibe == "INTIMACY": key = "whisper"
                    
                    possible = matrix.get(key, matrix.get("statement", ["..."]))
                    raw_response = random.choice(possible)
                    
                    attempts = 0
                    while raw_response in self.recent_responses and attempts < 5 and len(possible) > 1:
                        raw_response = random.choice(possible)
                        attempts += 1
                else:
                    raw_response = random.choice(["go on.", "i'm listening.", "tell me more."])

        self.recent_responses.append(raw_response)
        if len(self.recent_responses) > 10: self.recent_responses.pop(0)

        # 5. Pipeline Processing
        
        # A. Quick Detail Enhancement (Verification Layer) - Highest Priority
        enhanced_response = None
        try:
            enhanced_response = self.detail_enhancer.enhance_response(
                user_input,
                raw_response
            )
            if enhanced_response and enhanced_response != raw_response:
                raw_response = enhanced_response
        except Exception as e:
            print(f"Detail Enhancer Error: {e}")

        # Check if response is already structured (DeepSeek/Detailed)
        is_structured = raw_response.strip().startswith("#") or "Details:" in raw_response

        # B. Apply DeepSeek Emergency Fix (Directness Guarantee)
        # ONLY if not already structured and is a question
        if (is_question or is_direct_question) and not is_structured:
            try:
                raw_response = self.emergency_fix.fix_response(
                    user_input, 
                    raw_response, 
                    research_res
                )
            except Exception as e:
                print(f"DeepSeek Fix Error: {e}")


        # 6b. Apply Evolved Structure
        try:
            raw_response = self.evolved_builder.assemble(intel, raw_response, user_input)
        except Exception as e:
            print(f"Evolved Structure Error: {e}")

        # 6. Apply Voice (Cadence)
        # Check if structured response to avoid breaking formatting
        is_structured = "**" in raw_response or "Details:" in raw_response
        
        if is_structured:
            # Minimal cadence for structured responses
            final_response = raw_response
        else:
            final_response = self.cadence.apply_cadence(raw_response, emotion=active_vibe.lower())
        
        # 6. TTS Synthesis (Phase 3)
        audio_filename = f"resp_{int(time.time())}.wav"
        self.tts.synthesize(final_response, vibe=active_vibe, vibe_configs=self.vibe_states, output_file=audio_filename)
        
        # Mix Ambient (Phase 4)
        self.ambient_manager.mix_ambient(os.path.join("audio_out", audio_filename), active_vibe)
        
        # 7. Cinema Generation
        script_output = None
        if any(w in user_text for w in ["plot", "story", "movie", "script", "scene"]):
            movie_data = self.movie_gen.generate_movie(user_input, genre=active_vibe.capitalize())
            script_output = movie_data['script']
            # Optionally override final response if it's a dedicated plot request
            if len(user_text.split()) > 10:
                final_response = f"I've analyzed your premise. Here is a deconstructed view of the {active_vibe} arc.\n\n" + final_response

        # 8. Log and Learn
        self.processor.memory.log_interaction("agent", final_response)
        
        # 🧠 INTELLIGENCE UPGRADE: Update learning and context
        self.intelligence.process_response(user_input, final_response, active_vibe)
        
        # Prepare intelligence insights for frontend
        intel_insights = {
            "thought": thought_whisper,
            "intent": intel_intent,
            "role": intel_role,
            "confidence": round(intel_confidence, 2),
            "is_follow_up": is_follow_up,
            "user_facts": len(intel['memory']['user_facts_detected'])
        }
        
        return {
            "response": final_response,
            "dream": dream_fragment,
            "vibe_state": active_vibe,
            "vibe_color": vibe_config.get("color", "#6ee7b7"),
            "audio_url": f"/audio/{audio_filename}" if os.path.exists(os.path.join("audio_out", audio_filename)) else None,
            "code_update": script_output if script_output else "Lens Flare Initialized...",
            "shot_list": movie_data.get('shot_list') if 'movie_data' in locals() else [],
            "production_metadata": movie_data.get('metadata') if 'movie_data' in locals() else {},
            "character_dna": [c.get('dna') for c in movie_data.get('characters', [])] if 'movie_data' in locals() else [],
            "diagnostics": movie_data.get('diagnostics') if 'movie_data' in locals() else {},
            "production_metrics": movie_data.get('production_metrics') if 'movie_data' in locals() else {},
            "intelligence": intel_insights,
            "stats": intel['personality']['growth']
        }

agent = VibeServer()

@app.post("/chat")
async def chat(req: ChatRequest):
    return agent.get_response(req.message)

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r") as f:
        return f.read()

# 🧠 INTELLIGENCE UPGRADE: New API Endpoints

@app.get("/intelligence/stats")
async def intelligence_stats():
    """Get comprehensive intelligence system statistics"""
    return agent.intelligence.get_intelligence_stats()

@app.get("/intelligence/user")
async def user_profile():
    """Get what the agent has learned about the user"""
    return agent.intelligence.get_user_summary()

@app.get("/intelligence/learning")
async def learning_stats():
    """Get learning module statistics"""
    return agent.intelligence.learning_module.get_learning_stats()

@app.get("/intelligence/knowledge")
async def knowledge_stats():
    """Get knowledge graph statistics"""
    return agent.intelligence.knowledge_graph.get_graph_stats()

@app.post("/intelligence/feedback")
async def record_feedback(req: Request):
    """Record user feedback for learning"""
    data = await req.json()
    outcome = {
        "continued": data.get("continued", True),
        "feedback": data.get("feedback", "neutral"),
        "follow_up_type": data.get("follow_up_type", "")
    }
    # Get last interaction from memory
    recent = agent.intelligence.memory.get_recent_context(limit=1)
    if recent:
        effectiveness = agent.intelligence.record_outcome(
            recent[-1].get("content", "") if recent[-1].get("role") == "user" else "",
            data.get("response", ""),
            outcome
        )
        return {"recorded": True, "effectiveness": effectiveness}
    return {"recorded": False}

# 🆕 ADVANCED INTELLIGENCE ENDPOINTS

@app.get("/intelligence/emotional")
async def emotional_profile():
    """Get emotional intelligence analysis and user emotional profile"""
    return agent.intelligence.emotional_intelligence.get_emotional_profile()

@app.get("/intelligence/personality")
async def personality_state():
    """Get the agent's evolved personality state"""
    return agent.intelligence.personality.get_personality_summary()

@app.get("/intelligence/reflection")
async def get_reflection():
    """Get the agent's self-reflection on recent performance"""
    return agent.intelligence.get_periodic_reflection()

@app.get("/intelligence/creative/{concept_a}/{concept_b}")
async def creative_insight(concept_a: str, concept_b: str):
    """Generate a creative insight connecting two concepts"""
    return agent.intelligence.get_creative_insight(concept_a, concept_b)

@app.get("/intelligence/question/{topic}")
async def novel_question(topic: str):
    """Generate a novel, thought-provoking question about a topic"""
    return {"topic": topic, "question": agent.intelligence.get_novel_question(topic)}

@app.get("/intelligence/invitation/{topic}")
async def conversation_invitation(topic: str, style: str = "open"):
    """Get a conversation invitation for a topic"""
    return {"topic": topic, "style": style, "invitation": agent.intelligence.get_conversation_invitation(topic, style)}

@app.get("/intelligence/metacognition")
async def metacognition_state():
    """Get the agent's meta-cognitive self-awareness state"""
    return agent.intelligence.metacognition.get_self_awareness_summary()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

