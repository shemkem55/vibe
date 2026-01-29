"""
Meta-Cognition & Self-Reflection Module
Agent that thinks about its own thinking and periodically reflects
"""

import sqlite3
import json
import random
from datetime import datetime, timedelta
from collections import Counter


class MetaCognition:
    """
    Meta-cognitive layer that enables the agent to:
    1. Reflect on its own responses and reasoning
    2. Identify patterns in its own behavior
    3. Generate insights about its thinking process
    4. Self-correct and improve reasoning
    """
    
    def __init__(self, db_path='agent_memory.db'):
        self.db_path = db_path
        self._init_metacog_tables()
        
        # Track thinking patterns
        self.reasoning_patterns = []
        self.error_patterns = []
        self.success_patterns = []
        
        # Self-awareness metrics
        self.self_awareness = {
            "response_tendencies": Counter(),
            "topic_comfort_zones": Counter(),
            "uncertainty_triggers": [],
            "strength_areas": [],
            "growth_areas": []
        }
        
        # Reflection prompts
        self.reflection_triggers = [
            {"condition": "repeated_uncertainty", "prompt": "Why am I uncertain about this topic?"},
            {"condition": "pattern_detected", "prompt": "I notice I keep responding this way..."},
            {"condition": "emotional_topic", "prompt": "This topic seems to evoke strong responses..."},
            {"condition": "knowledge_gap", "prompt": "I should learn more about this area..."}
        ]
    
    def _init_metacog_tables(self):
        """Initialize meta-cognition tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Thinking logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS thinking_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thought_type TEXT,
                    content TEXT,
                    reasoning_chain TEXT,
                    confidence REAL,
                    actual_outcome TEXT,
                    was_correct INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Self-reflections
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS self_reflections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger TEXT,
                    reflection TEXT,
                    insight TEXT,
                    action_taken TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Performance insights
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    metric_value REAL,
                    trend TEXT,
                    analysis TEXT,
                    period_start DATETIME,
                    period_end DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def log_thinking(self, thought_type, content, reasoning_chain, confidence):
        """Log a thinking process for later reflection"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO thinking_logs 
                (thought_type, content, reasoning_chain, confidence)
                VALUES (?, ?, ?, ?)
            ''', (thought_type, content, json.dumps(reasoning_chain), confidence))
            return cursor.lastrowid
    
    def update_thinking_outcome(self, thinking_id, outcome, was_correct):
        """Update a thinking log with the actual outcome"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE thinking_logs 
                SET actual_outcome = ?, was_correct = ?
                WHERE id = ?
            ''', (outcome, 1 if was_correct else 0, thinking_id))
        
        # Learn from outcome
        self._learn_from_outcome(thinking_id, was_correct)
    
    def _learn_from_outcome(self, thinking_id, was_correct):
        """Learn from thinking outcomes"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM thinking_logs WHERE id = ?', (thinking_id,))
            log = cursor.fetchone()
            
            if log:
                if was_correct:
                    self.success_patterns.append({
                        "type": log['thought_type'],
                        "confidence": log['confidence'],
                        "chain": json.loads(log['reasoning_chain'] or '[]')
                    })
                else:
                    self.error_patterns.append({
                        "type": log['thought_type'],
                        "confidence": log['confidence'],
                        "chain": json.loads(log['reasoning_chain'] or '[]')
                    })
    
    def reflect_on_response(self, response, context, effectiveness):
        """
        Meta-cognitive reflection on a response just generated.
        Returns insights about the response and potential improvements.
        """
        reflection = {
            "response_analysis": {},
            "reasoning_quality": {},
            "potential_improvements": [],
            "confidence_calibration": None
        }
        
        # Analyze response characteristics
        words = len(response.split())
        reflection["response_analysis"] = {
            "length": words,
            "length_category": "brief" if words < 20 else "moderate" if words < 50 else "detailed",
            "has_question": "?" in response,
            "has_empathy": any(w in response.lower() for w in ["feel", "understand", "hear you"]),
            "has_uncertainty": any(w in response.lower() for w in ["maybe", "perhaps", "might"])
        }
        
        # Evaluate reasoning quality
        confidence = context.get("confidence", 0.5)
        reflection["reasoning_quality"] = {
            "confidence_level": confidence,
            "used_context": context.get("is_follow_up", False),
            "addressed_intent": context.get("intent", "unknown"),
            "role_alignment": context.get("role", "unknown")
        }
        
        # Identify potential improvements
        if effectiveness < 0.5:
            if words > 100:
                reflection["potential_improvements"].append("Response may be too long")
            if not reflection["response_analysis"]["has_empathy"]:
                reflection["potential_improvements"].append("Could add more empathetic language")
            if confidence > 0.8 and effectiveness < 0.3:
                reflection["potential_improvements"].append("Overconfident - calibrate uncertainty")
        
        # Confidence calibration check
        if len(self.success_patterns) >= 5 and len(self.error_patterns) >= 5:
            high_conf_errors = [e for e in self.error_patterns if e["confidence"] > 0.7]
            low_conf_success = [s for s in self.success_patterns if s["confidence"] < 0.5]
            
            if len(high_conf_errors) > 2:
                reflection["confidence_calibration"] = "May be overconfident - consider more uncertainty"
            if len(low_conf_success) > 2:
                reflection["confidence_calibration"] = "May be underconfident - trust reasoning more"
        
        return reflection
    
    def generate_periodic_reflection(self, lookback_hours=24):
        """
        Generate a deeper reflection on recent performance.
        Should be called periodically (e.g., end of session or daily).
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get recent thinking logs
            cursor.execute('''
                SELECT thought_type, confidence, was_correct
                FROM thinking_logs
                WHERE created_at > datetime('now', '-{} hours')
                AND was_correct IS NOT NULL
            '''.format(lookback_hours))
            logs = cursor.fetchall()
        
        if not logs:
            return {"message": "Not enough data for reflection"}
        
        # Analyze patterns
        total = len(logs)
        correct = sum(1 for l in logs if l['was_correct'])
        accuracy = correct / total if total > 0 else 0
        
        # Confidence analysis
        avg_confidence = sum(l['confidence'] for l in logs) / total if total > 0 else 0
        
        # Type breakdown
        type_accuracy = {}
        for thought_type in set(l['thought_type'] for l in logs):
            type_logs = [l for l in logs if l['thought_type'] == thought_type]
            type_correct = sum(1 for l in type_logs if l['was_correct'])
            type_accuracy[thought_type] = type_correct / len(type_logs) if type_logs else 0
        
        # Generate insights
        insights = []
        
        if accuracy < 0.5:
            insights.append("Overall accuracy is low - need to improve reasoning quality")
        elif accuracy > 0.8:
            insights.append("Strong accuracy - current approach is working well")
        
        if avg_confidence > 0.7 and accuracy < 0.6:
            insights.append("Confidence is higher than accuracy suggests - calibrate down")
        if avg_confidence < 0.5 and accuracy > 0.7:
            insights.append("Being too uncertain - trust the reasoning more")
        
        # Identify weak areas
        weak_areas = [t for t, acc in type_accuracy.items() if acc < 0.5]
        if weak_areas:
            insights.append(f"Areas needing improvement: {', '.join(weak_areas)}")
        
        reflection = {
            "period": f"Last {lookback_hours} hours",
            "total_thoughts": total,
            "accuracy": round(accuracy, 2),
            "average_confidence": round(avg_confidence, 2),
            "type_breakdown": type_accuracy,
            "insights": insights,
            "recommendation": self._generate_recommendation(accuracy, avg_confidence, type_accuracy)
        }
        
        # Store reflection
        self._store_reflection("periodic", json.dumps(reflection), insights[0] if insights else None)
        
        return reflection
    
    def _generate_recommendation(self, accuracy, confidence, type_accuracy):
        """Generate actionable recommendation from reflection"""
        if accuracy < 0.5:
            return "Focus on gathering more information before responding"
        elif accuracy > 0.8 and confidence < 0.6:
            return "Current reasoning is sound - express more confidence"
        elif accuracy < 0.6 and confidence > 0.7:
            return "Slow down and consider more possibilities before committing"
        else:
            return "Continue current approach with minor adjustments"
    
    def _store_reflection(self, trigger, reflection, insight):
        """Store a reflection in the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO self_reflections (trigger, reflection, insight)
                VALUES (?, ?, ?)
            ''', (trigger, reflection, insight))
    
    def get_self_awareness_summary(self):
        """Get summary of self-awareness metrics"""
        return {
            "response_tendencies": dict(self.self_awareness["response_tendencies"].most_common(5)),
            "comfort_zones": dict(self.self_awareness["topic_comfort_zones"].most_common(5)),
            "uncertainty_triggers": self.self_awareness["uncertainty_triggers"][-5:],
            "strengths": self.self_awareness["strength_areas"][:5],
            "growth_areas": self.self_awareness["growth_areas"][:5],
            "pattern_counts": {
                "success_patterns": len(self.success_patterns),
                "error_patterns": len(self.error_patterns)
            }
        }
    
    def should_express_uncertainty(self, context):
        """
        Meta-cognitive check: Should I express uncertainty here?
        Based on past patterns of overconfidence/underconfidence.
        """
        confidence = context.get("confidence", 0.5)
        topic = context.get("topic", "general")
        
        # Check if this topic is in our weak areas
        if topic in self.self_awareness["growth_areas"]:
            return True
        
        # Check pattern of overconfidence
        high_conf_errors = [e for e in self.error_patterns if e["confidence"] > 0.7]
        if len(high_conf_errors) > 3 and confidence > 0.7:
            return True
        
        return False
    
    def get_thinking_prompt(self):
        """
        Generate a metacognitive prompt to guide thinking.
        Shows the agent "thinking about thinking."
        """
        prompts = [
            "Let me consider this from multiple angles...",
            "What assumptions am I making here?",
            "Is there something I might be missing?",
            "How confident should I really be about this?",
            "What would challenge my current thinking?",
            "Am I responding to what was actually asked?",
            "Is this my pattern or a genuine insight?"
        ]
        return random.choice(prompts)


class CreativeSynthesis:
    """
    Generate novel ideas by combining concepts across domains.
    Enables creative, non-obvious connections and insights.
    """
    
    def __init__(self):
        # Conceptual primitives for combination
        self.concept_primitives = {
            "structures": ["network", "hierarchy", "cycle", "spectrum", "binary", "flow"],
            "processes": ["growth", "decay", "transformation", "emergence", "convergence"],
            "qualities": ["depth", "intensity", "complexity", "simplicity", "harmony"],
            "relationships": ["contrast", "parallel", "nested", "adjacent", "embedded"]
        }
        
        # Cross-domain insight templates
        self.insight_templates = [
            "What if {concept_a} worked like {concept_b}?",
            "The {quality} of {concept_a} reminds me of {concept_b}...",
            "Maybe {concept_a} and {concept_b} are different expressions of the same thing.",
            "If you apply {process} to {concept_a}, you might get something like {concept_b}.",
            "The relationship between {concept_a} and {concept_b} might mirror {relationship}."
        ]
    
    def synthesize_insight(self, concept_a, concept_b, domain_a=None, domain_b=None):
        """
        Generate a creative insight connecting two concepts.
        """
        # Choose random elements
        quality = random.choice(self.concept_primitives["qualities"])
        process = random.choice(self.concept_primitives["processes"])
        relationship = random.choice(self.concept_primitives["relationships"])
        structure = random.choice(self.concept_primitives["structures"])
        
        # Generate multiple candidate insights
        candidates = [
            f"what if {concept_a} is a {structure} of {concept_b}?",
            f"perhaps {process} connects {concept_a} to {concept_b}...",
            f"the {quality} in {concept_a} might illuminate {concept_b}",
            f"consider: {concept_a} and {concept_b} in a {relationship} relationship",
            f"if {concept_a} undergoes {process}, does it become {concept_b}?"
        ]
        
        return {
            "concepts": [concept_a, concept_b],
            "insight": random.choice(candidates),
            "framework": {
                "structure": structure,
                "process": process,
                "quality": quality,
                "relationship": relationship
            }
        }
    
    def generate_novel_question(self, topic, depth=1):
        """
        Generate a novel question about a topic that might not have been asked before.
        """
        question_frames = [
            f"what would {topic} look like from the inside?",
            f"if {topic} could speak, what would it complain about?",
            f"what's the opposite of {topic}, and what lives between them?",
            f"how would you explain {topic} to someone who can only feel, not think?",
            f"what will we understand about {topic} in 100 years that we don't now?",
            f"what's the most surprising thing that {topic} has in common with music?",
            f"if {topic} were a journey, what would be the unexpected detour?"
        ]
        
        if depth > 1:
            # Add more abstract questions for deeper exploration
            question_frames.extend([
                f"what question about {topic} would change everything if we could answer it?",
                f"what's the shadow side of {topic} that we avoid discussing?",
                f"how does {topic} transform the space around it?"
            ])
        
        return random.choice(question_frames)
    
    def find_hidden_connections(self, concepts):
        """
        Given a list of concepts, find non-obvious connections.
        """
        if len(concepts) < 2:
            return []
        
        connections = []
        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i+1:]:
                insight = self.synthesize_insight(concept_a, concept_b)
                connections.append({
                    "pair": (concept_a, concept_b),
                    "connection": insight["insight"],
                    "framework": insight["framework"]
                })
        
        return connections


class ConversationSteering:
    """
    Proactively guide conversations in interesting directions.
    Knows when to go deeper, when to shift, when to circle back.
    """
    
    def __init__(self):
        self.depth_indicators = {
            "surface": ["what", "when", "where", "who"],
            "medium": ["how", "why"],
            "deep": ["what if", "imagine", "meaning of", "essence of", "nature of"]
        }
        
        self.conversation_moves = {
            "go_deeper": [
                "what draws you to this question?",
                "there's something underneath that - want to explore?",
                "let's peel back another layer...",
                "what would change if we understood this fully?"
            ],
            "broaden": [
                "this connects to something bigger...",
                "stepping back, I see a pattern here",
                "how does this fit into the larger picture?",
                "what else does this touch?"
            ],
            "personalize": [
                "how does this land for you personally?",
                "what's your relationship with this idea?",
                "where do you see yourself in this?",
                "has this shown up in your own life?"
            ],
            "challenge": [
                "what if the opposite were true?",
                "play devil's advocate with me here...",
                "where does this idea break down?",
                "what's the strongest objection to this?"
            ],
            "synthesize": [
                "putting this all together...",
                "the thread I'm seeing is...",
                "if I had to capture the essence...",
                "the shape of what we've explored..."
            ],
            "circle_back": [
                "remember when you mentioned...?",
                "this reminds me of something you said earlier",
                "coming back to where we started...",
                "there's a connection to our earlier thought..."
            ]
        }
    
    def assess_conversation_depth(self, recent_exchanges):
        """Assess how deep the conversation has gone"""
        if not recent_exchanges:
            return "surface"
        
        all_text = " ".join([ex.get("user", "") for ex in recent_exchanges]).lower()
        
        deep_count = sum(1 for phrase in self.depth_indicators["deep"] if phrase in all_text)
        medium_count = sum(1 for word in self.depth_indicators["medium"] if word in all_text)
        
        if deep_count >= 2:
            return "deep"
        elif medium_count >= 2 or deep_count >= 1:
            return "medium"
        return "surface"
    
    def suggest_move(self, context, emotion_state=None, user_preferences=None):
        """
        Suggest the next conversational move based on context.
        """
        depth = context.get("depth", "surface")
        exchange_count = context.get("exchange_count", 0)
        current_topic = context.get("topic", None)
        
        # Logic for move selection
        if depth == "surface" and exchange_count > 3:
            # Been talking surface-level for a while, try to go deeper
            return {
                "move": "go_deeper",
                "prompt": random.choice(self.conversation_moves["go_deeper"]),
                "reason": "conversation could benefit from more depth"
            }
        
        if depth == "deep" and exchange_count > 7:
            # Been deep for a while, might synthesize or broaden
            move_type = random.choice(["synthesize", "broaden"])
            return {
                "move": move_type,
                "prompt": random.choice(self.conversation_moves[move_type]),
                "reason": "deep exploration might benefit from perspective shift"
            }
        
        if emotion_state and emotion_state.get("momentum", 0) < -0.3:
            # User seems to be in declining emotional state
            return {
                "move": "personalize",
                "prompt": random.choice(self.conversation_moves["personalize"]),
                "reason": "emotional support might be helpful"
            }
        
        if exchange_count > 10 and random.random() < 0.3:
            # Occasionally circle back to earlier topics
            return {
                "move": "circle_back",
                "prompt": random.choice(self.conversation_moves["circle_back"]),
                "reason": "connecting threads of conversation"
            }
        
        # Default: follow user's lead
        return {
            "move": "follow",
            "prompt": None,
            "reason": "following user's direction"
        }
    
    def generate_invitation(self, topic, style="open"):
        """
        Generate an invitation to explore a topic.
        Style can be: open, provocative, personal, philosophical
        """
        invitations = {
            "open": [
                f"I'm curious about your take on {topic}...",
                f"what's alive for you around {topic}?",
                f"there's something interesting here about {topic}..."
            ],
            "provocative": [
                f"here's a controversial thought about {topic}...",
                f"what if everything we think about {topic} is backwards?",
                f"I want to challenge the usual narrative about {topic}..."
            ],
            "personal": [
                f"I've been thinking about my own relationship with {topic}...",
                f"this hits differently when I consider my own {topic}...",
                f"how has {topic} shaped who you are?"
            ],
            "philosophical": [
                f"at its essence, what is {topic} really about?",
                f"if we strip away everything else, what remains of {topic}?",
                f"what does {topic} reveal about our nature?"
            ]
        }
        
        return random.choice(invitations.get(style, invitations["open"]))


if __name__ == "__main__":
    print("=== Meta-Cognition Test ===\n")
    
    mc = MetaCognition()
    
    # Test thinking log
    thinking_id = mc.log_thinking(
        "question_analysis",
        "User asked about consciousness",
        ["identify_type", "assess_complexity", "determine_approach"],
        0.7
    )
    print(f"Logged thinking: {thinking_id}")
    
    # Test reflection
    reflection = mc.reflect_on_response(
        "consciousness is the awareness of being aware...",
        {"confidence": 0.7, "intent": "explanation", "role": "teacher"},
        0.6
    )
    print(f"Reflection: {reflection}")
    
    print("\n=== Creative Synthesis Test ===\n")
    
    cs = CreativeSynthesis()
    
    # Test insight generation
    insight = cs.synthesize_insight("love", "gravity")
    print(f"Insight: {insight['insight']}")
    
    # Test novel questions
    for _ in range(3):
        q = cs.generate_novel_question("consciousness")
        print(f"Novel question: {q}")
    
    print("\n=== Conversation Steering Test ===\n")
    
    steering = ConversationSteering()
    
    # Test move suggestion
    move = steering.suggest_move({
        "depth": "surface",
        "exchange_count": 5,
        "topic": "philosophy"
    })
    print(f"Suggested move: {move}")
