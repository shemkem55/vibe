"""
Learning Module - Phase 3.1 Intelligence Upgrade
Continuous learning from interactions with pattern detection
"""

import sqlite3
import json
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict


class LearningModule:
    """Learn from every interaction and improve over time"""
    
    def __init__(self, db_path='agent_memory.db'):
        self.db_path = db_path
        self._init_learning_tables()
        self.pattern_detector = PatternDetector()
        
        # In-memory learning state
        self.session_lessons = []
        self.user_preferences = {}
        self.successful_patterns = []
    
    def _init_learning_tables(self):
        """Initialize learning-specific database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Learning log - tracks what worked and what didn't
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_pattern TEXT,
                    response_pattern TEXT,
                    effectiveness_score REAL,
                    user_feedback TEXT,
                    conversation_continued INTEGER,
                    response_length INTEGER,
                    vibe_state TEXT,
                    intent_type TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # User preference tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    preference_type TEXT,
                    preference_value TEXT,
                    confidence REAL,
                    sample_count INTEGER DEFAULT 1,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Successful response patterns
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS successful_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_pattern TEXT,
                    response_template TEXT,
                    success_rate REAL,
                    usage_count INTEGER DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def learn_from_interaction(self, user_input, agent_response, context, outcome=None):
        """
        Learn what works and what doesn't
        outcome: {"continued": bool, "feedback": str, "follow_up_type": str}
        """
        if outcome is None:
            outcome = {"continued": True, "feedback": "neutral"}
        
        # Extract patterns
        input_pattern = self._extract_pattern(user_input)
        response_pattern = self._extract_pattern(agent_response)
        
        # Calculate effectiveness
        effectiveness = self._calculate_effectiveness(
            user_input, agent_response, context, outcome
        )
        
        # Create lesson entry
        lesson = {
            "input_pattern": input_pattern,
            "response_pattern": response_pattern,
            "effectiveness": effectiveness,
            "outcome": outcome,
            "timestamp": datetime.now(),
            "vibe": context.get("vibe", "unknown"),
            "intent": context.get("intent", "unknown")
        }
        
        self.session_lessons.append(lesson)
        
        # Store in database
        self._save_lesson(lesson)
        
        # Learn from successful patterns
        if effectiveness > 0.7:
            self._save_successful_pattern(input_pattern, response_pattern, effectiveness)
        
        # Detect and update user preferences
        preferences = self.pattern_detector.detect_preferences(
            self.session_lessons[-10:]  # Look at last 10 interactions
        )
        self._update_preferences(preferences)
        
        return effectiveness
    
    def _extract_pattern(self, text):
        """Extract generalizable pattern from text"""
        text_lower = text.lower()
        
        # Identify key structural elements
        patterns = []
        
        # Question patterns
        if "?" in text:
            if text_lower.startswith(("what", "who", "where", "when")):
                patterns.append("wh_question")
            elif text_lower.startswith(("how", "why")):
                patterns.append("how_why_question")
            elif text_lower.startswith(("is", "are", "do", "does", "can")):
                patterns.append("yes_no_question")
            else:
                patterns.append("general_question")
        
        # Length pattern
        word_count = len(text.split())
        if word_count <= 5:
            patterns.append("brief")
        elif word_count <= 15:
            patterns.append("moderate")
        else:
            patterns.append("detailed")
        
        # Emotional markers
        if re.search(r"[!]{2,}|[?!]", text):
            patterns.append("emphatic")
        if re.search(r"\.{3,}|…", text):
            patterns.append("trailing")
        
        # Topic markers
        if re.search(r"\b(i feel|i think|i believe)\b", text_lower):
            patterns.append("personal_reflection")
        if re.search(r"\b(help|advice|suggest)\b", text_lower):
            patterns.append("seeking_help")
        
        return "+".join(patterns) if patterns else "general"
    
    def _calculate_effectiveness(self, user_input, agent_response, context, outcome):
        """How effective was this response?"""
        score = 0.5  # Base score
        
        # Conversation continuation is a strong signal
        if outcome.get("continued", True):
            score += 0.2
        else:
            score -= 0.2
        
        # Explicit feedback
        feedback = outcome.get("feedback", "neutral")
        if feedback in ["positive", "thanks", "good", "helpful"]:
            score += 0.2
        elif feedback in ["negative", "bad", "wrong", "confused"]:
            score -= 0.3
        
        # Response length appropriateness
        input_len = len(user_input.split())
        response_len = len(agent_response.split())
        
        # Generally, response should be somewhat proportional to input
        ratio = response_len / max(input_len, 1)
        if 0.5 <= ratio <= 3.0:
            score += 0.1  # Good balance
        elif ratio > 5.0 or ratio < 0.2:
            score -= 0.1  # Too verbose or too terse
        
        # Follow-up type
        follow_up = outcome.get("follow_up_type", "")
        if follow_up == "clarification":
            score -= 0.15  # User needed clarification = not clear enough
        elif follow_up == "expansion":
            score += 0.1  # User wanted more = interesting response
        
        return max(0.0, min(1.0, score))
    
    def _save_lesson(self, lesson):
        """Persist lesson to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO learning_log 
                (input_pattern, response_pattern, effectiveness_score, 
                 user_feedback, conversation_continued, response_length,
                 vibe_state, intent_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                lesson["input_pattern"],
                lesson["response_pattern"],
                lesson["effectiveness"],
                lesson["outcome"].get("feedback", "neutral"),
                1 if lesson["outcome"].get("continued", True) else 0,
                len(lesson["response_pattern"]),
                lesson["vibe"],
                lesson["intent"]
            ))
    
    def _save_successful_pattern(self, trigger, response, score):
        """Save patterns that worked well"""
        with sqlite3.connect(self.db_path) as conn:
            # Check if pattern exists
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id, success_rate, usage_count FROM successful_patterns WHERE trigger_pattern = ?',
                (trigger,)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update running average
                old_rate = existing[1]
                count = existing[2]
                new_rate = (old_rate * count + score) / (count + 1)
                conn.execute(
                    'UPDATE successful_patterns SET success_rate = ?, usage_count = usage_count + 1 WHERE id = ?',
                    (new_rate, existing[0])
                )
            else:
                # Insert new pattern
                conn.execute(
                    'INSERT INTO successful_patterns (trigger_pattern, response_template, success_rate) VALUES (?, ?, ?)',
                    (trigger, response, score)
                )
    
    def _update_preferences(self, preferences):
        """Update detected user preferences"""
        with sqlite3.connect(self.db_path) as conn:
            for pref_type, pref_value, confidence in preferences:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT id, confidence, sample_count FROM user_preferences WHERE preference_type = ?',
                    (pref_type,)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update with exponential moving average
                    old_conf = existing[1]
                    count = existing[2]
                    alpha = 0.3  # New observations have 30% weight
                    new_conf = alpha * confidence + (1 - alpha) * old_conf
                    conn.execute(
                        '''UPDATE user_preferences 
                           SET preference_value = ?, confidence = ?, sample_count = sample_count + 1, last_updated = ?
                           WHERE id = ?''',
                        (pref_value, new_conf, datetime.now(), existing[0])
                    )
                else:
                    conn.execute(
                        'INSERT INTO user_preferences (preference_type, preference_value, confidence) VALUES (?, ?, ?)',
                        (pref_type, pref_value, confidence)
                    )
    
    def get_learned_preferences(self):
        """Retrieve learned user preferences"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT preference_type, preference_value, confidence 
                   FROM user_preferences 
                   WHERE confidence > 0.5 AND sample_count >= 3
                   ORDER BY confidence DESC'''
            )
            return [dict(r) for r in cursor.fetchall()]
    
    def get_successful_patterns(self, trigger_pattern=None, limit=5):
        """Get patterns that have worked well"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if trigger_pattern:
                cursor.execute(
                    '''SELECT trigger_pattern, response_template, success_rate 
                       FROM successful_patterns 
                       WHERE trigger_pattern LIKE ? AND success_rate > 0.6
                       ORDER BY success_rate DESC LIMIT ?''',
                    (f"%{trigger_pattern}%", limit)
                )
            else:
                cursor.execute(
                    '''SELECT trigger_pattern, response_template, success_rate 
                       FROM successful_patterns 
                       WHERE success_rate > 0.6
                       ORDER BY success_rate DESC LIMIT ?''',
                    (limit,)
                )
            
            return [dict(r) for r in cursor.fetchall()]
    
    def suggest_improvement(self, current_response, context):
        """Suggest improvements based on learned patterns"""
        # Get user preferences
        preferences = self.get_learned_preferences()
        
        suggestions = []
        
        for pref in preferences:
            pref_type = pref["preference_type"]
            pref_value = pref["preference_value"]
            
            if pref_type == "response_length":
                current_len = len(current_response.split())
                if pref_value == "short" and current_len > 30:
                    suggestions.append("Consider shortening response - user prefers brevity")
                elif pref_value == "detailed" and current_len < 20:
                    suggestions.append("Consider elaborating - user prefers detailed responses")
            
            elif pref_type == "formality":
                if pref_value == "casual" and not any(c in current_response for c in ["...", "hm", "yeah"]):
                    suggestions.append("Consider more casual tone")
            
            elif pref_type == "explanation_style":
                if pref_value == "examples" and "for example" not in current_response.lower():
                    suggestions.append("User learns better with examples")
        
        return suggestions
    
    def get_learning_stats(self):
        """Get statistics about learning progress"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total interactions learned
            cursor.execute('SELECT COUNT(*) FROM learning_log')
            total = cursor.fetchone()[0]
            
            # Average effectiveness
            cursor.execute('SELECT AVG(effectiveness_score) FROM learning_log')
            avg_effectiveness = cursor.fetchone()[0] or 0.5
            
            # Recent improvement (last 50 vs previous 50)
            cursor.execute('''
                SELECT AVG(effectiveness_score) FROM (
                    SELECT effectiveness_score FROM learning_log ORDER BY id DESC LIMIT 50
                )
            ''')
            recent_avg = cursor.fetchone()[0] or 0.5
            
            cursor.execute('''
                SELECT AVG(effectiveness_score) FROM (
                    SELECT effectiveness_score FROM learning_log ORDER BY id DESC LIMIT 100 OFFSET 50
                )
            ''')
            previous_avg = cursor.fetchone()[0] or 0.5
            
            improvement = recent_avg - previous_avg if total >= 100 else 0
            
            return {
                "total_interactions": total,
                "average_effectiveness": round(avg_effectiveness, 3),
                "recent_average": round(recent_avg, 3),
                "improvement_trend": round(improvement, 3),
                "preferences_learned": len(self.get_learned_preferences()),
                "successful_patterns": len(self.get_successful_patterns(limit=100))
            }

class ContinuousLearningLoop:
    """🆕 Real-time feedback and online adaptation loop"""
    
    def __init__(self, learner: LearningModule):
        self.learner = learner
        self.adaptation_history = []
        self.performance_window = [] # Recent effectiveness scores
        
    def process_explicit_feedback(self, feedback_text: str, rating: float):
        """Process direct user ratings and comments"""
        # Logic to extract actionable fixes from feedback
        pass

    def online_adaptation(self, current_vibe: str, context: dict):
        """Adjust model parameters or strategy based on recent performance"""
        stats = self.learner.get_learning_stats()
        adaptation = {
            "temperature_delta": 0.0,
            "top_p_delta": 0.0,
            "strategy_shift": None
        }
        
        # If performance is declining, become more cautious
        if stats["improvement_trend"] < -0.1:
            adaptation["temperature_delta"] = -0.1
            adaptation["strategy_shift"] = "CAUTIOUS_PRECISION"
        # If performance is high, allow more creativity
        elif stats["recent_average"] > 0.8:
            adaptation["temperature_delta"] = 0.05
            adaptation["strategy_shift"] = "CREATIVE_FLOW"
            
        return adaptation

    def track_session_performance(self, score: float):
        """Track effectiveness within a single session"""
        self.performance_window.append(score)
        if len(self.performance_window) > 5:
            self.performance_window.pop(0)
            
        return sum(self.performance_window) / len(self.performance_window)


class PatternDetector:
    """Detect patterns in user behavior and conversation"""
    
    def __init__(self):
        self.response_length_history = []
        self.topic_history = []
        self.time_patterns = defaultdict(list)
    
    def detect_preferences(self, recent_lessons):
        """Detect user preferences from recent interactions"""
        if len(recent_lessons) < 3:
            return []
        
        preferences = []
        
        # Detect response length preference
        length_pref = self._detect_length_preference(recent_lessons)
        if length_pref:
            preferences.append(length_pref)
        
        # Detect formality preference
        formality_pref = self._detect_formality_preference(recent_lessons)
        if formality_pref:
            preferences.append(formality_pref)
        
        # Detect topic patterns
        topic_pref = self._detect_topic_patterns(recent_lessons)
        if topic_pref:
            preferences.append(topic_pref)
        
        return preferences
    
    def _detect_length_preference(self, lessons):
        """Detect if user prefers short or long responses"""
        # Get effectiveness by response length
        short_scores = []
        long_scores = []
        
        for lesson in lessons:
            response_pattern = lesson.get("response_pattern", "")
            effectiveness = lesson.get("effectiveness", 0.5)
            
            if "brief" in response_pattern:
                short_scores.append(effectiveness)
            elif "detailed" in response_pattern:
                long_scores.append(effectiveness)
        
        if short_scores and long_scores:
            short_avg = sum(short_scores) / len(short_scores)
            long_avg = sum(long_scores) / len(long_scores)
            
            if short_avg > long_avg + 0.1:
                return ("response_length", "short", short_avg)
            elif long_avg > short_avg + 0.1:
                return ("response_length", "detailed", long_avg)
        
        return None
    
    def _detect_formality_preference(self, lessons):
        """Detect preferred formality level"""
        # Based on input patterns
        casual_markers = 0
        formal_markers = 0
        
        for lesson in lessons:
            pattern = lesson.get("input_pattern", "")
            if "emphatic" in pattern or "trailing" in pattern:
                casual_markers += 1
            if "detailed" in pattern and "how_why_question" in pattern:
                formal_markers += 1
        
        total = casual_markers + formal_markers
        if total >= 3:
            if casual_markers > formal_markers:
                return ("formality", "casual", casual_markers / total)
            elif formal_markers > casual_markers:
                return ("formality", "formal", formal_markers / total)
        
        return None
    
    def _detect_topic_patterns(self, lessons):
        """Detect recurring topic interests"""
        intents = [l.get("intent", "unknown") for l in lessons]
        
        if len(intents) >= 5:
            intent_counts = Counter(intents)
            top_intent, count = intent_counts.most_common(1)[0]
            
            if count >= 3 and top_intent != "unknown":
                return ("preferred_intent", top_intent, count / len(intents))
        
        return None
    
    def predict_next_topic(self, current_topic, history):
        """Predict what the user might ask about next"""
        # Build transition probabilities
        transitions = defaultdict(Counter)
        
        for i in range(len(history) - 1):
            current = history[i].get("intent", "unknown")
            next_topic = history[i + 1].get("intent", "unknown")
            transitions[current][next_topic] += 1
        
        if current_topic in transitions:
            predictions = transitions[current_topic].most_common(3)
            return [topic for topic, _ in predictions]
        
        return []


if __name__ == "__main__":
    # Test the learning module
    learner = LearningModule()
    
    print("=== Learning Module Test ===\n")
    
    # Simulate some interactions
    test_interactions = [
        ("What is the meaning of life?", "that's a deep question... perhaps it's about finding your own purpose.", {"continued": True, "feedback": "positive"}),
        ("Tell me more", "meaning can come from connections, creativity, or contribution to others.", {"continued": True, "feedback": "neutral"}),
        ("Thanks!", "you're welcome. always here to explore ideas with you.", {"continued": False, "feedback": "positive"}),
        ("Why is the sky blue?", "it's due to Rayleigh scattering - shorter wavelengths scatter more in the atmosphere.", {"continued": True, "feedback": "positive"}),
        ("Explain more simply", "sunlight bends in air - blue bends the most so we see blue sky.", {"continued": True, "feedback": "positive"})
    ]
    
    for user_input, response, outcome in test_interactions:
        context = {"vibe": "CURIOSITY", "intent": "explanation"}
        score = learner.learn_from_interaction(user_input, response, context, outcome)
        print(f"Input: '{user_input[:40]}...'")
        print(f"Effectiveness: {score:.2f}")
        print("-" * 40)
    
    print("\n=== Learning Stats ===")
    stats = learner.get_learning_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Learned Preferences ===")
    prefs = learner.get_learned_preferences()
    for p in prefs:
        print(f"- {p['preference_type']}: {p['preference_value']} (confidence: {p['confidence']:.2f})")
