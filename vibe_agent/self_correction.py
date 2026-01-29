"""
Self-Correction Engine - Phase 3 Intelligence Upgrade
Enables the agent to review, critique, and refine its own outputs before they reach the user.
"""

from typing import Dict, List, Any, Optional
import re

class SelfCorrectionEngine:
    """Agentic layer for reviewing and correcting internal outputs"""
    
    def __init__(self, llm=None):
        self.llm = llm
        self.critique_criteria = [
            "factual_consistency",
            "logical_coherence",
            "persona_alignment",
            "emotional_resonance",
            "brevity_and_clarity"
        ]

    def review_response(self, response: str, intel_packet: Dict) -> Dict[str, Any]:
        """Review a candidate response and provide a critique"""
        critique = {
            "score": 1.0,
            "issues": [],
            "should_regenerate": False,
            "refinements": []
        }
        
        # 1. Check for persona alignment
        active_persona = intel_packet.get("personality", {}).get("active_persona", "THE_COMPANION")
        personality_summary = intel_packet.get("personality", {}).get("summary", {})
        
        # Simple rule-based persona checks
        if active_persona == "THE_ANALYST" and len(response.split()) > 150:
            critique["issues"].append("Response is too verbose for An Analyst profile.")
            critique["score"] -= 0.2
            
        # 2. Check for Hallucinations (Simulated)
        # If response mentions facts not in knowledge connections or reasoning
        if intel_packet.get("reasoning", {}).get("confidence", 1.0) < 0.4:
            if not any(word in response.lower() for word in ["perhaps", "might", "uncertain", "believe"]):
                critique["issues"].append("Low confidence response missing uncertainty markers.")
                critique["score"] -= 0.3
                
        # 3. Emotional Resonance
        empathetic_res = intel_packet.get("emotional", {}).get("empathetic_response")
        if empathetic_res and len(response) < 20:
             critique["issues"].append("Response may be too brief to convey empathy.")
             critique["score"] -= 0.1

        if critique["score"] < 0.6:
            critique["should_regenerate"] = True
            
        return critique

    def self_correct(self, response: str, critique: Dict, intel_packet: Dict) -> str:
        """Apply corrections to the response based on critique"""
        if not critique["issues"]:
            return response
            
        corrected_response = response
        
        # If we have an LLM, we can do a proper refinement call
        if self.llm:
            refinement_prompt = f"""
            Original Response: {response}
            Critique: {', '.join(critique["issues"])}
            Intelligence Context: {intel_packet.get('understanding', {}).get('intent')}
            
            Please rewrite the response to address the issues in the critique while maintaining the core meaning.
            """
            # corrected_response = self.llm.chat([{"role": "user", "content": refinement_prompt}], profile="primary")
            # For now, simulate with markers or simple edits
            pass
            
        # Simple rule-based correction (fallback)
        if "Low confidence response missing uncertainty markers" in str(critique["issues"]):
            corrected_response = "I suspect that " + response[0].lower() + response[1:] + " (though I am still processing this)."
            
        return corrected_response

class AutonomousGoalManager:
    """Manages internal agentic goals and sub-tasks"""
    
    def __init__(self):
        self.active_goals = []
        self.completed_goals = []

    def set_goal(self, goal_type: str, metadata: Dict):
        self.active_goals.append({"type": goal_type, "metadata": metadata, "status": "pending"})

    def get_next_subtask(self) -> Optional[Dict]:
        if not self.active_goals:
            return None
        return self.active_goals[0]
