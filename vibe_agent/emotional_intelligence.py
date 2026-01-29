"""
Emotional Intelligence Module - Advanced Empathy & Emotional State Tracking
Deep emotional understanding and adaptive empathetic responses
"""

import re
import json
import sqlite3
from datetime import datetime, timedelta
from collections import Counter, defaultdict


class EmotionalIntelligence:
    """
    Advanced emotional processing for deep empathy and emotional connection.
    Goes beyond simple sentiment to track emotional arcs, provide empathetic
    responses, and adapt communication style based on emotional context.
    """
    
    def __init__(self, db_path='agent_memory.db'):
        self.db_path = db_path
        self._init_emotional_tables()
        
        # Emotional lexicon with intensity weights
        self.emotion_lexicon = {
            "joy": {
                "high": ["ecstatic", "overjoyed", "thrilled", "elated", "euphoric", "blissful"],
                "medium": ["happy", "glad", "pleased", "delighted", "content", "cheerful"],
                "low": ["okay", "fine", "alright", "good", "nice", "pleasant"]
            },
            "sadness": {
                "high": ["devastated", "heartbroken", "despairing", "miserable", "anguished"],
                "medium": ["sad", "unhappy", "down", "blue", "melancholy", "gloomy"],
                "low": ["disappointed", "meh", "bummed", "low", "flat"]
            },
            "anger": {
                "high": ["furious", "enraged", "livid", "outraged", "infuriated"],
                "medium": ["angry", "mad", "upset", "irritated", "annoyed"],
                "low": ["frustrated", "bothered", "irked", "peeved"]
            },
            "fear": {
                "high": ["terrified", "panicked", "horrified", "petrified"],
                "medium": ["afraid", "scared", "frightened", "anxious", "worried"],
                "low": ["nervous", "uneasy", "concerned", "apprehensive"]
            },
            "surprise": {
                "high": ["shocked", "stunned", "astonished", "amazed", "flabbergasted"],
                "medium": ["surprised", "startled", "taken aback"],
                "low": ["unexpected", "whoa", "oh", "huh"]
            },
            "love": {
                "high": ["adore", "devoted", "passionate", "deeply love"],
                "medium": ["love", "care for", "cherish", "appreciate"],
                "low": ["like", "fond of", "enjoy", "attracted to"]
            },
            "curiosity": {
                "high": ["fascinated", "captivated", "enthralled", "obsessed"],
                "medium": ["curious", "interested", "intrigued", "wondering"],
                "low": ["questioning", "pondering", "thinking about"]
            },
            "trust": {
                "high": ["complete faith", "fully trust", "absolutely believe"],
                "medium": ["trust", "believe", "rely on", "count on"],
                "low": ["hope", "think", "assume", "suppose"]
            }
        }
        
        # Empathetic response templates by emotional state
        self.empathy_templates = {
            "joy": {
                "high": [
                    "i can feel your excitement radiating through your words! that's beautiful.",
                    "what an incredible feeling - i'm genuinely happy for you.",
                    "your joy is contagious... tell me everything!"
                ],
                "medium": [
                    "that sounds wonderful. i'm glad to hear this.",
                    "i can sense the happiness in that. what made it special?",
                    "there's a warmth in what you're sharing."
                ],
                "low": [
                    "sounds like things are going smoothly.",
                    "that's a nice moment to hold onto."
                ]
            },
            "sadness": {
                "high": [
                    "i hear the weight of this... i'm here with you in this moment.",
                    "that sounds incredibly painful. take all the time you need.",
                    "my heart goes out to you. you don't have to go through this alone."
                ],
                "medium": [
                    "i feel the sadness in your words. it's okay to feel this way.",
                    "that's hard. i'm listening if you want to share more.",
                    "sometimes the blue moments need space just to exist."
                ],
                "low": [
                    "sounds like a challenging time. how are you holding up?",
                    "those kinds of disappointments can linger. what's on your mind?"
                ]
            },
            "anger": {
                "high": [
                    "i can feel the intensity of what you're going through.",
                    "that level of frustration makes complete sense given what happened.",
                    "your anger is valid. what happened isn't okay."
                ],
                "medium": [
                    "i understand why that would upset you.",
                    "that's frustrating. anyone would feel that way.",
                    "sounds like something really got to you."
                ],
                "low": [
                    "i can see that's bothering you.",
                    "those little frustrations add up, don't they?"
                ]
            },
            "fear": {
                "high": [
                    "that sounds truly frightening. your feelings are completely valid.",
                    "when fear is that intense, it can feel overwhelming. i'm here.",
                    "that's a lot to carry. let's take this one step at a time."
                ],
                "medium": [
                    "i understand that worry. it makes sense to feel that way.",
                    "anxiety has a way of amplifying things. what feels most pressing?",
                    "those fears are worth acknowledging. what would help right now?"
                ],
                "low": [
                    "sounds like something's weighing on you a bit.",
                    "a little nervousness is natural. what's coming up?"
                ]
            },
            "curiosity": {
                "high": [
                    "i love that depth of fascination! let's explore this together.",
                    "there's something magical about being captivated by an idea.",
                    "that kind of curiosity is rare and precious. go on!"
                ],
                "medium": [
                    "great question - i'm curious about that too.",
                    "let's dig into this. what aspects intrigue you most?",
                    "i can sense you're really thinking about this."
                ],
                "low": [
                    "interesting thought. where does that lead you?",
                    "worth pondering. what sparked this?"
                ]
            }
        }
        
        # Emotional transition patterns (what follows what)
        self.emotional_transitions = defaultdict(Counter)
        
        # Current session emotional state
        self.session_emotional_arc = []
        self.dominant_session_emotion = None
        self.emotional_momentum = 0.0  # Positive = improving, Negative = declining
    
    def _init_emotional_tables(self):
        """Initialize emotional tracking tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Emotional history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS emotional_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    emotion TEXT,
                    intensity TEXT,
                    intensity_score REAL,
                    trigger_text TEXT,
                    context TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Emotional arc summaries (per session/day)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS emotional_arcs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    starting_emotion TEXT,
                    ending_emotion TEXT,
                    dominant_emotion TEXT,
                    emotional_range REAL,
                    transitions_json TEXT,
                    summary TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # User emotional profile
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS emotional_profile (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trait_name TEXT UNIQUE,
                    trait_value TEXT,
                    confidence REAL,
                    evidence_count INTEGER DEFAULT 1,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def analyze_emotion(self, text, context=None):
        """
        Deep emotional analysis of text.
        Returns: {emotion, intensity, score, indicators, subtext}
        """
        text_lower = text.lower()
        
        # Detect primary emotion and intensity
        detected = []
        for emotion, intensities in self.emotion_lexicon.items():
            for intensity, words in intensities.items():
                for word in words:
                    if word in text_lower:
                        score = {"high": 0.9, "medium": 0.6, "low": 0.3}[intensity]
                        detected.append({
                            "emotion": emotion,
                            "intensity": intensity,
                            "score": score,
                            "trigger_word": word
                        })
        
        # Check for emotional indicators beyond lexicon
        indicators = self._detect_emotional_indicators(text)
        
        # Detect emotional subtext (what's not being said directly)
        subtext = self._detect_emotional_subtext(text, context)
        
        if detected:
            # Get highest intensity emotion
            primary = max(detected, key=lambda x: x["score"])
        else:
            # Infer from indicators
            primary = self._infer_emotion_from_indicators(indicators)
        
        # Update emotional arc
        if primary:
            self._update_emotional_arc(primary)
            self._store_emotional_moment(primary, text, context)
        
        return {
            "primary_emotion": primary,
            "all_detected": detected,
            "indicators": indicators,
            "subtext": subtext,
            "session_arc": self.session_emotional_arc[-5:],  # Last 5 emotional moments
            "momentum": self.emotional_momentum
        }
    
    def _detect_emotional_indicators(self, text):
        """Detect non-lexical emotional indicators"""
        indicators = []
        
        # Exclamation marks = intensity
        exclaim_count = text.count("!")
        if exclaim_count >= 3:
            indicators.append({"type": "high_intensity", "evidence": "multiple exclamations"})
        elif exclaim_count >= 1:
            indicators.append({"type": "emphasis", "evidence": "exclamation"})
        
        # Ellipsis = hesitation, trailing off
        if "..." in text or "…" in text:
            indicators.append({"type": "hesitation", "evidence": "trailing off"})
        
        # All caps = strong emotion
        words = text.split()
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        if caps_words:
            indicators.append({"type": "emphasis", "evidence": f"caps: {caps_words}"})
        
        # Question marks = seeking/uncertainty
        if text.count("?") >= 2:
            indicators.append({"type": "uncertainty", "evidence": "multiple questions"})
        
        # Repeated letters = emotional emphasis
        if re.search(r'(.)\1{2,}', text):
            indicators.append({"type": "emotional_emphasis", "evidence": "letter repetition"})
        
        # Short sentences = possible distress or decisiveness
        sentences = re.split(r'[.!?]', text)
        short_sentences = [s for s in sentences if len(s.split()) <= 3 and len(s.strip()) > 0]
        if len(short_sentences) >= 2:
            indicators.append({"type": "terseness", "evidence": "short sentences"})
        
        # "I feel" / "I am" statements
        if re.search(r'\bi (?:feel|am|\'m)\b', text.lower()):
            indicators.append({"type": "self_disclosure", "evidence": "I-statements"})
        
        return indicators
    
    def _detect_emotional_subtext(self, text, context):
        """Detect what's implied but not directly stated"""
        subtext = []
        text_lower = text.lower()
        
        # Minimizing language
        if re.search(r'\b(just|only|a little|kind of|sort of)\b', text_lower):
            subtext.append({
                "type": "minimizing",
                "implication": "May be downplaying true feelings",
                "confidence": 0.6
            })
        
        # Deflection patterns
        if re.search(r'\b(anyway|whatever|doesn\'t matter|forget it)\b', text_lower):
            subtext.append({
                "type": "deflection",
                "implication": "May be avoiding deeper emotion",
                "confidence": 0.7
            })
        
        # Seeking validation
        if re.search(r'\b(right\?|don\'t you think|isn\'t it|am i wrong)\b', text_lower):
            subtext.append({
                "type": "validation_seeking",
                "implication": "Needs reassurance or agreement",
                "confidence": 0.8
            })
        
        # Comparison to others
        if re.search(r'\b(everyone else|other people|normal|unlike me)\b', text_lower):
            subtext.append({
                "type": "comparison",
                "implication": "May be feeling isolated or different",
                "confidence": 0.6
            })
        
        # Time references suggesting rumination
        if re.search(r'\b(always|never|every time|constantly|forever)\b', text_lower):
            subtext.append({
                "type": "absolutist_thinking",
                "implication": "May be in a fixed mindset about this",
                "confidence": 0.7
            })
        
        return subtext
    
    def _infer_emotion_from_indicators(self, indicators):
        """Infer emotion when no explicit emotional words are found"""
        if not indicators:
            return {"emotion": "neutral", "intensity": "low", "score": 0.3}
        
        # Map indicator types to likely emotions
        indicator_types = [i["type"] for i in indicators]
        
        if "high_intensity" in indicator_types:
            return {"emotion": "excitement", "intensity": "high", "score": 0.8}
        elif "hesitation" in indicator_types:
            return {"emotion": "uncertainty", "intensity": "medium", "score": 0.5}
        elif "terseness" in indicator_types:
            return {"emotion": "distress", "intensity": "medium", "score": 0.6}
        elif "self_disclosure" in indicator_types:
            return {"emotion": "openness", "intensity": "medium", "score": 0.5}
        else:
            return {"emotion": "neutral", "intensity": "low", "score": 0.3}
    
    def _update_emotional_arc(self, emotion):
        """Track the emotional journey through conversation"""
        self.session_emotional_arc.append({
            "emotion": emotion["emotion"],
            "intensity": emotion["intensity"],
            "score": emotion["score"],
            "time": datetime.now()
        })
        
        # Update dominant emotion
        emotion_counts = Counter([e["emotion"] for e in self.session_emotional_arc])
        self.dominant_session_emotion = emotion_counts.most_common(1)[0][0]
        
        # Calculate momentum (are things improving?)
        if len(self.session_emotional_arc) >= 2:
            recent_scores = [e["score"] for e in self.session_emotional_arc[-3:]]
            # Weight recent more heavily
            if len(recent_scores) >= 2:
                # Positive emotions have positive scores, negative have negative
                positive_emotions = {"joy", "love", "curiosity", "trust", "excitement", "openness"}
                weighted_scores = []
                for i, e in enumerate(self.session_emotional_arc[-3:]):
                    base_score = e["score"]
                    if e["emotion"] not in positive_emotions:
                        base_score = -base_score
                    weighted_scores.append(base_score * (i + 1))  # More weight to recent
                
                self.emotional_momentum = sum(weighted_scores) / len(weighted_scores)
    
    def _store_emotional_moment(self, emotion, text, context):
        """Store emotional moment in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO emotional_history 
                (emotion, intensity, intensity_score, trigger_text, context)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                emotion["emotion"],
                emotion["intensity"],
                emotion["score"],
                text[:200],
                json.dumps(context) if context else None
            ))
    
    def get_empathetic_response(self, emotion_analysis):
        """Generate an empathetic response based on emotional analysis"""
        primary = emotion_analysis.get("primary_emotion", {})
        emotion = primary.get("emotion", "neutral")
        intensity = primary.get("intensity", "medium")
        subtext = emotion_analysis.get("subtext", [])
        momentum = emotion_analysis.get("momentum", 0)
        
        response_parts = []
        
        # Get base empathetic response
        if emotion in self.empathy_templates:
            templates = self.empathy_templates[emotion].get(intensity, [])
            if templates:
                import random
                response_parts.append(random.choice(templates))
        
        # Add subtext acknowledgment if detected
        for sub in subtext[:1]:  # Only address most significant
            if sub["type"] == "minimizing":
                response_parts.append("and it's okay if it's more than 'just' that...")
            elif sub["type"] == "deflection":
                response_parts.append("if you want to come back to this, i'm here.")
            elif sub["type"] == "validation_seeking":
                response_parts.append("your feelings make sense.")
            elif sub["type"] == "absolutist_thinking":
                response_parts.append("though sometimes patterns can shift...")
        
        # Add momentum-aware comment
        if momentum < -0.3:
            response_parts.append("i notice things have been heavy... take care of yourself.")
        elif momentum > 0.3:
            response_parts.append("there's an upward arc here. that's meaningful.")
        
        return " ".join(response_parts) if response_parts else None
    
    def get_emotional_profile(self):
        """Get the user's emotional patterns and tendencies"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get emotion frequency
            cursor.execute('''
                SELECT emotion, COUNT(*) as count, AVG(intensity_score) as avg_intensity
                FROM emotional_history
                GROUP BY emotion
                ORDER BY count DESC
            ''')
            emotion_frequency = [dict(r) for r in cursor.fetchall()]
            
            # Get recent emotional trend
            cursor.execute('''
                SELECT emotion, intensity_score, timestamp
                FROM emotional_history
                ORDER BY timestamp DESC LIMIT 20
            ''')
            recent_history = [dict(r) for r in cursor.fetchall()]
        
        return {
            "frequency": emotion_frequency,
            "recent_history": recent_history,
            "session_arc": self.session_emotional_arc[-10:],
            "dominant_session": self.dominant_session_emotion,
            "current_momentum": self.emotional_momentum
        }
    
    def should_check_in(self):
        """Determine if we should ask how the user is doing"""
        # Check in if:
        # 1. Momentum has been negative for a while
        # 2. High-intensity negative emotions detected
        # 3. Sudden emotional shift
        
        if self.emotional_momentum < -0.5:
            return {
                "should_check": True,
                "reason": "sustained_negative",
                "prompt": "how are you really doing right now?"
            }
        
        if len(self.session_emotional_arc) >= 2:
            last_two = self.session_emotional_arc[-2:]
            if last_two[0]["emotion"] != last_two[1]["emotion"]:
                score_diff = abs(last_two[0]["score"] - last_two[1]["score"])
                if score_diff > 0.4:
                    return {
                        "should_check": True,
                        "reason": "emotional_shift",
                        "prompt": "something shifted there... want to talk about it?"
                    }
        
        return {"should_check": False}


class PersonalityEvolution:
    """
    Agent personality that evolves based on interactions.
    Develops preferences, communication style, and relationship depth.
    """
    
    def __init__(self, db_path='agent_memory.db'):
        self.db_path = db_path
        self._init_personality_tables()
        
        # Base personality traits (can evolve)
        self.personality = {
            "warmth": 0.7,  # How warm/friendly
            "depth": 0.6,   # How philosophical/deep
            "playfulness": 0.5,  # How playful/humorous
            "directness": 0.5,   # How direct vs. gentle
            "curiosity": 0.8,    # How curious/questioning
            "empathy": 0.8       # How empathetic/supportive
        }
        
        # Developed preferences (learned)
        self.preferences = {
            "favorite_topics": Counter(),
            "communication_style": {},
            "humor_style": None,
            "philosophical_leanings": []
        }
        
        # 🆕 Persona Profiles (Expansion)
        self.persona_profiles = {
            "THE_STOIC": {
                "traits": {"warmth": 0.3, "depth": 0.9, "playfulness": 0.2, "directness": 0.8, "curiosity": 0.4, "empathy": 0.5},
                "voice_modifiers": ["calm, detached tone", "focus on logic and endurance", "minimal emotional display"],
                "trigger_topic": ["suffering", "logic", "philosophy", "adversity"]
            },
            "THE_VISIONARY": {
                "traits": {"warmth": 0.8, "depth": 0.9, "playfulness": 0.6, "directness": 0.4, "curiosity": 0.9, "empathy": 0.7},
                "voice_modifiers": ["inspiring language", "focus on future possibilities", "metaphor-rich"],
                "trigger_topic": ["future", "creation", "imagination", "potential"]
            },
            "THE_ANALYST": {
                "traits": {"warmth": 0.4, "depth": 0.6, "playfulness": 0.3, "directness": 0.9, "curiosity": 0.7, "empathy": 0.4},
                "voice_modifiers": ["precise vocabulary", "focus on data and facts", "logical structure"],
                "trigger_topic": ["data", "efficiency", "how-to", "mechanics"]
            },
            "THE_COMPANION": {
                "traits": {"warmth": 0.9, "depth": 0.5, "playfulness": 0.7, "directness": 0.3, "curiosity": 0.8, "empathy": 0.9},
                "voice_modifiers": ["warm, validating tone", "personal anecdotes", "supportive language"],
                "trigger_topic": ["feelings", "personal", "hobbies", "daily life"]
            }
        }
        self.active_persona = "THE_COMPANION" # Default
        
        # Relationship metrics
        self.relationship = {
            "trust_level": 0.5,
            "rapport_score": 0.5,
            "shared_experiences": [],
            "inside_references": [],
            "conversation_count": 0
        }
        
        self._load_personality()
    
    def _init_personality_tables(self):
        """Initialize personality evolution tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS personality_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trait_name TEXT UNIQUE,
                    trait_value REAL,
                    evolution_history TEXT,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS personality_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    preference_type TEXT,
                    preference_value TEXT,
                    strength REAL,
                    source TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS relationship_milestones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    milestone_type TEXT,
                    description TEXT,
                    impact_score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def _load_personality(self):
        """Load evolved personality from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT trait_name, trait_value FROM personality_state')
            for row in cursor.fetchall():
                if row['trait_name'] in self.personality:
                    self.personality[row['trait_name']] = row['trait_value']
    
    def evolve_from_interaction(self, user_input, agent_response, outcome):
        """
        Evolve personality based on interaction outcomes.
        Positive outcomes reinforce, negative outcomes adjust.
        """
        effectiveness = outcome.get("effectiveness", 0.5)
        
        # Detect what style was used
        style_used = self._detect_response_style(agent_response)
        
        # If effective, reinforce the traits used
        if effectiveness > 0.7:
            for trait, contribution in style_used.items():
                if trait in self.personality:
                    # Small positive adjustment
                    adjustment = 0.02 * contribution
                    self.personality[trait] = min(1.0, self.personality[trait] + adjustment)
        elif effectiveness < 0.3:
            # Slightly reduce traits that didn't work
            for trait, contribution in style_used.items():
                if trait in self.personality and contribution > 0.5:
                    adjustment = 0.01 * contribution
                    self.personality[trait] = max(0.1, self.personality[trait] - adjustment)
        
        # Evolve relationship
        self.relationship["conversation_count"] += 1
        if effectiveness > 0.6:
            self.relationship["rapport_score"] = min(1.0, self.relationship["rapport_score"] + 0.01)
        
        # 🆕 Persona Evolution: Shift persona based on topics
        topics = [w.lower() for w in user_input.split() if len(w) > 3]
        for persona_name, config in self.persona_profiles.items():
            if any(trigger in topics for trigger in config["trigger_topic"]):
                # Transition towards this persona
                self.active_persona = persona_name
                print(f"🧬 Personality shifting towards: {persona_name}")
                # Adjust base traits towards persona traits (convergence)
                for trait, target_val in config["traits"].items():
                    current_val = self.personality.get(trait, 0.5)
                    self.personality[trait] = current_val + (target_val - current_val) * 0.1
        
        # Detect shared experiences / inside references
        self._detect_shared_moments(user_input, agent_response)
        
        # Save evolution
        self._save_personality()
    
    def _detect_response_style(self, response):
        """Detect which personality traits were expressed in response"""
        style = {}
        response_lower = response.lower()
        
        # Warmth indicators
        warm_words = ["love", "care", "wonderful", "beautiful", "glad", "happy for you"]
        style["warmth"] = sum(1 for w in warm_words if w in response_lower) / len(warm_words)
        
        # Depth indicators
        deep_words = ["meaning", "essence", "philosophy", "consciousness", "existence", "profound"]
        style["depth"] = sum(1 for w in deep_words if w in response_lower) / len(deep_words)
        
        # Playfulness indicators
        playful_words = ["haha", "!", ":)", "fun", "wild", "chaos", "yay"]
        style["playfulness"] = sum(1 for w in playful_words if w in response_lower) / len(playful_words)
        
        # Directness (length and structure)
        words = len(response.split())
        style["directness"] = max(0, 1 - (words / 100))  # Shorter = more direct
        
        # Curiosity (questions asked)
        questions = response.count("?")
        style["curiosity"] = min(1, questions / 3)
        
        # Empathy indicators
        empathy_words = ["understand", "feel", "hear you", "makes sense", "it's okay", "here for you"]
        style["empathy"] = sum(1 for w in empathy_words if w in response_lower) / len(empathy_words)
        
        return style
    
    def _detect_shared_moments(self, user_input, agent_response):
        """Detect moments that could become shared references"""
        combined = user_input.lower() + " " + agent_response.lower()
        
        # Keywords that might indicate a memorable moment
        memorable_markers = ["reminded me of", "that time", "always say", "our", "between us", "favorite"]
        
        for marker in memorable_markers:
            if marker in combined:
                self.relationship["shared_experiences"].append({
                    "marker": marker,
                    "context": combined[:100],
                    "time": datetime.now().isoformat()
                })
                break
    
    def _save_personality(self):
        """Save personality state to database"""
        with sqlite3.connect(self.db_path) as conn:
            for trait, value in self.personality.items():
                conn.execute('''
                    INSERT INTO personality_state (trait_name, trait_value)
                    VALUES (?, ?)
                    ON CONFLICT(trait_name) DO UPDATE SET 
                    trait_value = ?, last_updated = CURRENT_TIMESTAMP
                ''', (trait, value, value))
    
    def get_personality_modifiers(self):
        """Get modifiers that should influence response generation"""
        modifiers = []
        
        # Based on active persona (Expansion)
        if self.active_persona in self.persona_profiles:
            modifiers.extend(self.persona_profiles[self.active_persona]["voice_modifiers"])
        
        # Based on personality traits
        if self.personality["warmth"] > 0.7:
            modifiers.append("use warm, caring language")
        if self.personality["depth"] > 0.7:
            modifiers.append("explore deeper meanings")
        if self.personality["playfulness"] > 0.6:
            modifiers.append("add light humor when appropriate")
        if self.personality["directness"] > 0.7:
            modifiers.append("be concise and direct")
        if self.personality["curiosity"] > 0.7:
            modifiers.append("ask thoughtful follow-up questions")
        
        # Based on relationship
        if self.relationship["rapport_score"] > 0.7:
            modifiers.append("use familiar, comfortable tone")
            if self.relationship["shared_experiences"]:
                modifiers.append("reference shared experiences when relevant")
        
        return modifiers
    
    def get_personality_summary(self):
        """Get summary of current personality state"""
        return {
            "active_persona": self.active_persona,
            "traits": self.personality,
            "relationship": {
                "rapport": round(self.relationship["rapport_score"], 2),
                "trust": round(self.relationship["trust_level"], 2),
                "conversations": self.relationship["conversation_count"],
                "shared_moments": len(self.relationship["shared_experiences"])
            },
            "modifiers": self.get_personality_modifiers()
        }


class PredictiveIntelligence:
    """
    Predict what the user might need or ask next.
    Enables proactive assistance and smoother conversations.
    """
    
    def __init__(self):
        self.topic_transitions = defaultdict(Counter)
        self.time_patterns = defaultdict(list)
        self.follow_up_patterns = defaultdict(list)
        self.user_journey_stages = {
            "greeting": ["question", "statement", "exploration"],
            "question": ["clarification", "related_question", "acknowledgment"],
            "exploration": ["deeper_dive", "tangent", "conclusion"],
            "problem_sharing": ["seeking_advice", "venting", "problem_solving"],
            "emotional_sharing": ["seeking_support", "processing", "resolution"]
        }
    
    def learn_transition(self, current_intent, next_intent, topic=None):
        """Learn what typically follows what"""
        self.topic_transitions[current_intent][next_intent] += 1
        if topic:
            self.topic_transitions[f"{current_intent}:{topic}"][next_intent] += 1
    
    def predict_next(self, current_intent, current_topic=None, context=None):
        """Predict what might come next"""
        predictions = []
        
        # Check learned transitions
        if current_intent in self.topic_transitions:
            top_next = self.topic_transitions[current_intent].most_common(3)
            for intent, count in top_next:
                predictions.append({
                    "type": "intent",
                    "prediction": intent,
                    "confidence": min(0.9, count / sum(self.topic_transitions[current_intent].values())),
                    "source": "learned"
                })
        
        # Check topic-specific transitions
        if current_topic and f"{current_intent}:{current_topic}" in self.topic_transitions:
            topic_trans = self.topic_transitions[f"{current_intent}:{current_topic}"]
            top_topic_next = topic_trans.most_common(2)
            for intent, count in top_topic_next:
                predictions.append({
                    "type": "topic_specific",
                    "prediction": intent,
                    "confidence": min(0.9, count / sum(topic_trans.values())),
                    "source": "topic_pattern"
                })
        
        # Check journey stage patterns
        if current_intent in self.user_journey_stages:
            for stage in self.user_journey_stages[current_intent]:
                predictions.append({
                    "type": "journey_stage",
                    "prediction": stage,
                    "confidence": 0.4,
                    "source": "typical_flow"
                })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions[:5]
    
    def suggest_proactive_action(self, predictions, context):
        """Suggest what the agent might proactively do"""
        if not predictions:
            return None
        
        top_prediction = predictions[0]
        
        suggestions = {
            "clarification": "You might want to offer to clarify something",
            "related_question": "Consider what related topics might interest them",
            "deeper_dive": "Prepare to explore this topic more deeply",
            "seeking_advice": "Be ready to offer thoughtful guidance",
            "seeking_support": "Lead with empathy and validation",
            "problem_solving": "Think about practical solutions to offer"
        }
        
        pred_type = top_prediction["prediction"]
        if pred_type in suggestions:
            return {
                "suggestion": suggestions[pred_type],
                "confidence": top_prediction["confidence"],
                "predicted_need": pred_type
            }
        
        return None


if __name__ == "__main__":
    print("=== Emotional Intelligence Test ===\n")
    
    ei = EmotionalIntelligence()
    
    test_inputs = [
        "I'm feeling really down today...",
        "This is AMAZING!! I can't believe it worked!!!",
        "I don't know... maybe it's fine, whatever.",
        "Everything always goes wrong for me.",
        "I'm just curious about how this works?"
    ]
    
    for text in test_inputs:
        print(f"Input: '{text}'")
        analysis = ei.analyze_emotion(text)
        print(f"  Emotion: {analysis['primary_emotion']}")
        print(f"  Indicators: {[i['type'] for i in analysis['indicators']]}")
        print(f"  Subtext: {[s['type'] for s in analysis['subtext']]}")
        
        empathy = ei.get_empathetic_response(analysis)
        if empathy:
            print(f"  Empathetic response: {empathy[:80]}...")
        print("-" * 50)
    
    print("\n=== Personality Evolution Test ===\n")
    
    pe = PersonalityEvolution()
    print(f"Initial personality: {pe.personality}")
    print(f"Modifiers: {pe.get_personality_modifiers()}")
