"""
Vibe Agent - High Intelligence CLI
The terminal interface for your upgraded AI companion.
Integrates DeepSeek-quality responses, emotional intelligence, and deep research.
"""

import sys
import time
import random
import json
from processor import InputProcessor
from cadence import CadenceController
from thought_stream import ThoughtStream
from audio_listener import AudioListener
from research_engine import ResearchTrigger, ResearchEngine
from research_response import ResearchResponseBuilder
from direct_answer_generator import EmergencyResponseFix, DirectAnswerCore
from emergency_detail_upgrade import QuickDetailEnhancer
from context_manager import ContextManager, QuestionClassifier
from intelligence_orchestrator import IntelligenceOrchestrator
from response_architect import EvolvedResponseArchitect

class VibeAgent:
    def __init__(self):
        print("🧠 Initializing Vibe Agent Intelligence...")
        self.processor = InputProcessor()
        self.cadence = CadenceController()
        self.thought_stream = ThoughtStream()
        self.listener = AudioListener()
        self.name = self.processor.essence.get("name", "Vibe")
        
        # Core Intelligence Systems
        self.intelligence = IntelligenceOrchestrator()
        self.context_manager = ContextManager(window_size=10)
        
        # Research & Response Systems
        self.research_trigger = ResearchTrigger()
        self.research_engine = ResearchEngine()
        self.research_builder = ResearchResponseBuilder()
        self.question_classifier = QuestionClassifier()
        self.emergency_fix = EmergencyResponseFix()
        self.detail_enhancer = QuickDetailEnhancer()
        self.evolved_builder = EvolvedResponseArchitect()
        
        # Fallback knowledge
        self.knowledge = self._load_knowledge()
        
        print("✅ Intelligence Systems Online.")
        
    def _load_knowledge(self):
        try:
            with open("knowledge.json", "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def generate_response(self, user_input, context):
        """
        Generate a high-intelligence response using the full pipeline.
        """
        user_text = user_input.lower()
        active_vibe = context['user_analysis'].get('detected_emotion', 'REFLECTION').upper()
        
        # 1. Intelligence Processing
        intel = self.intelligence.process_input(user_input, active_vibe)
        
        # 2. Check for Direct Question / Research Need
        question_data = self.question_classifier.classify_question(user_input)
        is_direct = question_data.get("needs_direct_answer", False)
        is_question = "?" in user_input or is_direct
        
        research_needed = self.research_trigger.needs_research(user_input)
        
        # Explicitly research if it's a question
        if is_question and not research_needed:
             research_needed = {"type": "general_question", "depth": "quick"}
             
        # 3. Execute Research if needed
        research_res = None
        if research_needed:
            print(f"\n🔍 Researching: {user_input[:40]}...")
            research_res = self.research_engine.research(user_input, depth=research_needed["depth"])
        
        # 4. Build Response
        raw_response = ""
        
        if research_res and research_res.get("summary"):
            # Use DeepSeek Quality Builder
            raw_response = self.research_builder.build_response(
                research_res,
                vibe=active_vibe,
                direct_mode=is_direct,
                question=user_input,
                deepseek_mode=True
            )
        elif intel['reasoning'].get('llm_output'):
            # Use Local LLM output
            raw_response = intel['reasoning']['llm_output']
        else:
            # Fallback Logic
            if is_question:
                # Use intelligence context or clarification
                if intel['understanding']['should_clarify']:
                    raw_response = "i want to make sure i understand... could you tell me more about what you're looking for?"
                else:
                    raw_response = "i couldn't find specific data on that. could you rephrase?"
            else:
                # Check static knowledge
                for topic, responses in self.knowledge.items():
                    if topic in user_text:
                        raw_response = random.choice(responses)
                        break
                
                if not raw_response:
                    # Generic conversational response
                    raw_response = "i'm listening. tell me more."

        # 5. Apply DeepSeek Emergency Fix (Directness Guarantee)
        if is_question:
            try:
                raw_response = self.emergency_fix.fix_response(user_input, raw_response, research_res)
                # 5b. Apply Quick Detail Enhancement
                raw_response = self.detail_enhancer.enhance_response(user_input, raw_response)
            except Exception:
                pass

        # 6. Apply Voice/Cadence (unless perfectly structured)
        if "**" not in raw_response and "Details:" not in raw_response:
            final_response = self.cadence.apply_cadence(raw_response, emotion=active_vibe.lower())
        else:
            final_response = raw_response
            
        # 7. Upgrade to Evolved Structure (Final Assembly)
        try:
            final_response = self.evolved_builder.assemble(intel, final_response, user_input)
        except Exception as e:
            print(f"Warning: Evolved assembly failed: {e}")
            
        # 🎬 Execute Autonomous Actions (Expansion)
        self.intelligence.execute_autonomous_actions(intel)
        
        # Log performance
        proc_time = intel.get("production", {}).get("processing_time_ms", 0)
        if proc_time > 500:
             print(f"   ⏱️ Heavy cognitive load: {proc_time}ms")
            
        return final_response, intel

    def chat_loop(self):
        print(f"\n🌌 {self.name} is waking up...")
        time.sleep(1.0)
        print(f"[{self.name}]: hey. i'm listening. what's on your mind?\n")
        
        while True:
            try:
                # Audio Input Option
                prompt = "You (Type or Press Enter to speak): " if self.listener.is_available() else "You: "
                user_input = input(prompt)
                
                # Check for voice trigger (empty input + available listener)
                if not user_input.strip() and self.listener.is_available():
                    user_input = self.listener.listen()
                    if not user_input:
                        continue # Start loop again to re-prompt
                        
                if not user_input or not user_input.strip():
                    continue
                
                # 1. Process Context
                context = self.processor.process(user_input)
                intent = context['user_analysis']['intent']
                
                # 2. Simulate Thinking (Visual Indicator)
                thought_seed = self.thought_stream.generate_seed(length=random.randint(4, 6))
                print(f"[{self.name} thinking: {thought_seed}]", end="", flush=True)
                
                # Delay simulation based on complexity
                delay = 1.0 + (len(user_input) * 0.01)
                for _ in range(4):
                    time.sleep(delay / 4)
                    print(".", end="", flush=True)
                print("\r" + " " * 80 + "\r", end="", flush=True) 

                # 3. Generate Response
                response, intel = self.generate_response(user_input, context)
                
                # 4. Feed back to thought stream
                self.thought_stream.learn_from_text(user_input)
                
                # 5. Output with typewriter effect
                print(f"[{self.name}]: ", end="")
                for char in response:
                    print(char, end="", flush=True)
                    # Faster typing for long responses
                    sleep_time = 0.005 if len(response) > 200 else 0.015
                    time.sleep(sleep_time)
                print("\n")
                
                # 6. Post-Response Processing
                self.intelligence.process_response(user_input, response)
                
                if intent == "farewell":
                    break
                    
            except KeyboardInterrupt:
                print(f"\n[{self.name}]: fading out... save travels.")
                break
            except Exception as e:
                print(f"\n[System Error]: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    agent = VibeAgent()
    agent.chat_loop()

