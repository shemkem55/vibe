"""
Enhanced Emotional Intelligence v2 - Advanced Emotional Processing
Implements sophisticated emotion detection, empathy modeling, and emotional memory
"""

import json
import re
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import sqlite3

class EmotionIntensity(Enum):
    VERY_LOW = (0.0, 0.2, "barely noticeable")
    LOW = (0.2, 0.4, "mild")
    MODERATE = (0.4, 0.6, "noticeable")
    HIGH = (0.6, 0.8, "strong")
    VERY_HIGH = (0.8, 1.0, "intense")

class EmotionalState(Enum):
    STABLE = "stable"
    IMPROVING = "improving"
    DECLINING = "declining"
    VOLATILE = "volatile"
    TRANSITIONING = "transitioning"

@dataclass
class EmotionVector:
    """Multi-dimensional emotion representation"""
    joy: float = 0.0
    sadness: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    disgust: float = 0.0
    trust: float = 0.0
    anticipation: float = 0.0
    
    def magnitude(self) -> float:
        """Calculate emotional magnitude"""
        return math.sqrt(sum(v**2 for v in asdict(self).values()))
    
    def dominant_emotion(self) -> Tuple[str, float]:
        """Get the dominant emotion and its intensity"""
        emotions = asdict(self)
        dominant = max(emotions.items(), key=lambda x: x[1])
        return dominant
    
    def emotional_complexity(self) -> float:
        """Measure emotional complexity (how many emotions are active)"""
        emotions = asdict(self)
        active_emotions = sum(1 for v in emotions.values() if v > 0.1)
        return active_emotions / len(emotions)

@dataclass
class EmotionalContext:
    """Context surrounding an emotional state"""
    triggers: List[str]
    situation: str
    social_context: str
    temporal_context: str
    intensity_factors: List[str]
    coping_mechanisms: List[str]

@dataclass
class EmotionalMemory:
    """Memory of an emotional experience"""
    timestamp: datetime
    emotion_vector: EmotionVector
    context: EmotionalContext
    user_response: str
    agent_response: str
    effectiveness_score: float
    learned_patterns: List[str]

class AdvancedEmotionalIntelligence:
    """
    Enhanced emotional intelligence with sophisticated emotion modeling,
    empathy simulation, and emotional memory
    """
    
    def __init__(self, db_path='agent_memory.db'):
        self.db_path = db_path
        self.emotion_lexicon = self._build_enhanced_lexicon()
        self.empathy_templates = self._load_empathy_templates()
        self.emotional_patterns = self._load_emotional_patterns()
        self.cultural_contexts = self._load_cultural_contexts()
        self.emotional_memory = []
        self._init_emotional_memory_db()
        
    def _build_enhanced_lexicon(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive emotion lexicon with context sensitivity"""
        return {
            "joy": {
                "keywords": [
                    "happy", "joyful", "excited", "thrilled", "delighted", "elated",
                    "cheerful", "glad", "pleased", "content", "satisfied", "euphoric",
                    "ecstatic", "overjoyed", "blissful", "radiant", "beaming"
                ],
                "contextual_modifiers": {
                    "achievement": 1.3,  # Joy from achievement is stronger
                    "social": 1.2,       # Social joy is enhanced
                    "surprise": 1.4,     # Unexpected joy is more intense
                    "relief": 0.9        # Relief-joy is more subdued
                },
                "intensity_markers": {
                    "very": 1.5, "extremely": 1.8, "incredibly": 1.7,
                    "somewhat": 0.7, "a bit": 0.5, "slightly": 0.4
                },
                "physiological_markers": [
                    "smiling", "laughing", "grinning", "beaming", "glowing"
                ]
            },
            "sadness": {
                "keywords": [
                    "sad", "depressed", "melancholy", "sorrowful", "mournful",
                    "dejected", "despondent", "downhearted", "blue", "gloomy",
                    "miserable", "heartbroken", "devastated", "grief", "bereaved"
                ],
                "contextual_modifiers": {
                    "loss": 1.5,
                    "disappointment": 1.2,
                    "loneliness": 1.3,
                    "nostalgia": 0.8
                },
                "intensity_markers": {
                    "deeply": 1.6, "profoundly": 1.7, "utterly": 1.8,
                    "somewhat": 0.7, "a little": 0.5
                },
                "physiological_markers": [
                    "crying", "tears", "weeping", "sobbing", "sighing"
                ]
            },
            "anger": {
                "keywords": [
                    "angry", "furious", "enraged", "livid", "irate", "incensed",
                    "outraged", "indignant", "resentful", "hostile", "irritated",
                    "annoyed", "frustrated", "aggravated", "infuriated"
                ],
                "contextual_modifiers": {
                    "injustice": 1.4,
                    "betrayal": 1.5,
                    "frustration": 1.2,
                    "righteous": 1.3
                },
                "intensity_markers": {
                    "absolutely": 1.7, "completely": 1.6, "totally": 1.5,
                    "somewhat": 0.7, "mildly": 0.5
                },
                "physiological_markers": [
                    "shouting", "yelling", "clenching", "fuming", "seething"
                ]
            },
            "fear": {
                "keywords": [
                    "afraid", "scared", "terrified", "frightened", "anxious",
                    "worried", "nervous", "apprehensive", "panicked", "alarmed",
                    "concerned", "uneasy", "dread", "phobic", "paranoid"
                ],
                "contextual_modifiers": {
                    "unknown": 1.3,
                    "threat": 1.5,
                    "anticipation": 1.2,
                    "social": 1.1
                },
                "intensity_markers": {
                    "absolutely": 1.8, "completely": 1.7, "totally": 1.6,
                    "somewhat": 0.7, "a bit": 0.5
                },
                "physiological_markers": [
                    "trembling", "shaking", "sweating", "heart racing", "pale"
                ]
            },
            "surprise": {
                "keywords": [
                    "surprised", "shocked", "astonished", "amazed", "stunned",
                    "bewildered", "startled", "flabbergasted", "dumbfounded",
                    "speechless", "unexpected", "sudden", "wow"
                ],
                "contextual_modifiers": {
                    "positive": 1.2,
                    "negative": 1.3,
                    "neutral": 1.0
                },
                "intensity_markers": {
                    "completely": 1.6, "totally": 1.5, "absolutely": 1.7,
                    "somewhat": 0.7, "a little": 0.5
                },
                "physiological_markers": [
                    "gasping", "eyes wide", "jaw dropped", "frozen"
                ]
            },
            "trust": {
                "keywords": [
                    "trust", "confident", "secure", "safe", "comfortable",
                    "reliable", "dependable", "faithful", "loyal", "assured",
                    "certain", "convinced", "believe", "faith"
                ],
                "contextual_modifiers": {
                    "relationship": 1.3,
                    "experience": 1.2,
                    "intuition": 1.1
                },
                "intensity_markers": {
                    "completely": 1.6, "absolutely": 1.7, "totally": 1.5,
                    "somewhat": 0.7, "partially": 0.6
                },
                "physiological_markers": [
                    "relaxed", "calm", "open", "leaning in"
                ]
            },
            "anticipation": {
                "keywords": [
                    "excited", "eager", "looking forward", "anticipating",
                    "expecting", "hopeful", "optimistic", "enthusiastic",
                    "can't wait", "thrilled", "pumped", "ready"
                ],
                "contextual_modifiers": {
                    "positive_outcome": 1.3,
                    "event": 1.2,
                    "change": 1.1
                },
                "intensity_markers": {
                    "really": 1.4, "so": 1.3, "very": 1.5,
                    "somewhat": 0.7, "a bit": 0.5
                },
                "physiological_markers": [
                    "energetic", "restless", "animated", "bright eyes"
                ]
            },
            "disgust": {
                "keywords": [
                    "disgusted", "revolted", "repulsed", "sickened", "nauseated",
                    "appalled", "horrified", "repelled", "grossed out", "offended"
                ],
                "contextual_modifiers": {
                    "moral": 1.4,
                    "physical": 1.2,
                    "social": 1.3
                },
                "intensity_markers": {
                    "absolutely": 1.7, "completely": 1.6, "totally": 1.5,
                    "somewhat": 0.7, "a little": 0.5
                },
                "physiological_markers": [
                    "grimacing", "turning away", "covering face", "recoiling"
                ]
            }
        }
    
    def _load_empathy_templates(self) -> Dict[str, List[str]]:
        """Load empathy response templates for different emotional states"""
        return {
            "validation": [
                "That sounds really {emotion_adjective}. It makes complete sense that you'd feel this way.",
                "I can understand why you're feeling {emotion}. That's a natural response to what you're going through.",
                "Your feelings are completely valid. Anyone in your situation would likely feel {emotion}.",
                "It's okay to feel {emotion} about this. These feelings are important and deserve acknowledgment."
            ],
            "normalization": [
                "Many people experience {emotion} in situations like this. You're not alone in feeling this way.",
                "Feeling {emotion} is a common and healthy response to what you're experiencing.",
                "It's completely normal to feel {emotion} when dealing with {situation_type}."
            ],
            "support": [
                "I'm here with you through this {emotion}. You don't have to face this alone.",
                "Thank you for sharing these {emotion} feelings with me. That takes courage.",
                "I want you to know that your {emotion} is heard and understood."
            ],
            "gentle_reframe": [
                "While this {emotion} is difficult, it shows how much you care about {subject}.",
                "This {emotion} you're feeling is actually a sign of your {positive_quality}.",
                "Even in this {emotion}, I can see your strength and resilience."
            ],
            "future_focus": [
                "These {emotion} feelings won't last forever. You've gotten through difficult times before.",
                "While you're feeling {emotion} now, there are brighter moments ahead.",
                "This {emotion} is temporary, but your ability to overcome challenges is lasting."
            ]
        }
    
    def _load_emotional_patterns(self) -> Dict[str, Any]:
        """Load patterns for emotional state transitions and triggers"""
        return {
            "transition_patterns": {
                "sadness_to_anger": {
                    "triggers": ["injustice", "betrayal", "helplessness"],
                    "probability": 0.3,
                    "duration": "minutes to hours"
                },
                "fear_to_anger": {
                    "triggers": ["threat", "cornered", "protective"],
                    "probability": 0.4,
                    "duration": "seconds to minutes"
                },
                "anger_to_sadness": {
                    "triggers": ["exhaustion", "realization", "regret"],
                    "probability": 0.5,
                    "duration": "hours to days"
                },
                "surprise_to_joy": {
                    "triggers": ["positive_outcome", "gift", "achievement"],
                    "probability": 0.6,
                    "duration": "immediate"
                },
                "surprise_to_fear": {
                    "triggers": ["threat", "unknown", "sudden_change"],
                    "probability": 0.4,
                    "duration": "immediate"
                }
            },
            "emotional_cycles": {
                "grief": ["denial", "anger", "bargaining", "depression", "acceptance"],
                "stress_response": ["alarm", "resistance", "exhaustion"],
                "excitement": ["anticipation", "peak", "satisfaction", "reflection"]
            },
            "intensity_factors": {
                "personal_significance": 1.5,
                "unexpectedness": 1.3,
                "social_context": 1.2,
                "physical_state": 1.1,
                "time_pressure": 1.2,
                "control_level": -0.3  # More control = less intense negative emotions
            }
        }
    
    def _load_cultural_contexts(self) -> Dict[str, Any]:
        """Load cultural context modifiers for emotion expression"""
        return {
            "expression_norms": {
                "individualistic": {
                    "emotional_expression": 1.2,
                    "direct_communication": 1.3,
                    "personal_focus": 1.4
                },
                "collectivistic": {
                    "emotional_restraint": 1.2,
                    "harmony_focus": 1.3,
                    "group_consideration": 1.4
                }
            },
            "emotion_values": {
                "western": {
                    "happiness": 1.3,
                    "individual_achievement": 1.2,
                    "self_expression": 1.3
                },
                "eastern": {
                    "harmony": 1.3,
                    "respect": 1.2,
                    "collective_wellbeing": 1.3
                }
            }
        }
    
    def _init_emotional_memory_db(self):
        """Initialize emotional memory database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotional_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                emotion_vector TEXT,
                context TEXT,
                user_response TEXT,
                agent_response TEXT,
                effectiveness_score REAL,
                learned_patterns TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def analyze_emotional_content(self, text: str, context: Dict[str, Any] = None) -> EmotionVector:
        """Analyze emotional content with enhanced context sensitivity"""
        
        text_lower = text.lower()
        emotion_scores = {emotion: 0.0 for emotion in EmotionVector.__annotations__.keys()}
        
        # Analyze each emotion
        for emotion_name, emotion_data in self.emotion_lexicon.items():
            if emotion_name not in emotion_scores:
                continue
                
            base_score = 0.0
            
            # Keyword matching with intensity
            for keyword in emotion_data["keywords"]:
                if keyword in text_lower:
                    base_score += 0.1
                    
                    # Apply intensity markers
                    for marker, multiplier in emotion_data["intensity_markers"].items():
                        if marker in text_lower and keyword in text_lower:
                            base_score *= multiplier
                            break
            
            # Apply contextual modifiers
            if context:
                for context_type, modifier in emotion_data["contextual_modifiers"].items():
                    if context.get(context_type, False):
                        base_score *= modifier
            
            # Check for physiological markers
            for marker in emotion_data["physiological_markers"]:
                if marker in text_lower:
                    base_score += 0.05
            
            # Normalize and cap at 1.0
            emotion_scores[emotion_name] = min(1.0, base_score)
        
        # Apply emotional pattern adjustments
        emotion_scores = self._apply_emotional_patterns(emotion_scores, text, context)
        
        return EmotionVector(**emotion_scores)
    
    def _apply_emotional_patterns(self, emotion_scores: Dict[str, float], text: str, context: Dict[str, Any] = None) -> Dict[str, float]:
        """Apply emotional pattern knowledge to adjust scores"""
        
        # Check for emotional transitions
        dominant_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        
        if len(dominant_emotions) >= 2:
            primary, secondary = dominant_emotions[0], dominant_emotions[1]
            
            # Look for transition patterns
            transition_key = f"{secondary[0]}_to_{primary[0]}"
            if transition_key in self.emotional_patterns["transition_patterns"]:
                pattern = self.emotional_patterns["transition_patterns"][transition_key]
                
                # Check if triggers are present
                text_lower = text.lower()
                trigger_present = any(trigger in text_lower for trigger in pattern["triggers"])
                
                if trigger_present:
                    # Boost primary emotion based on transition probability
                    emotion_scores[primary[0]] *= (1 + pattern["probability"])
        
        # Apply intensity factors
        if context:
            for factor, multiplier in self.emotional_patterns["intensity_factors"].items():
                if context.get(factor, False):
                    # Apply to all emotions
                    for emotion in emotion_scores:
                        if multiplier > 0:
                            emotion_scores[emotion] *= (1 + multiplier)
                        else:
                            emotion_scores[emotion] *= (1 + abs(multiplier))
        
        return emotion_scores
    
    def generate_empathetic_response(self, emotion_vector: EmotionVector, context: EmotionalContext = None) -> Dict[str, Any]:
        """Generate empathetic response based on emotional analysis"""
        
        dominant_emotion, intensity = emotion_vector.dominant_emotion()
        
        # Determine intensity level
        intensity_level = self._categorize_intensity(intensity)
        
        # Select appropriate empathy template
        template_type = self._select_empathy_template(dominant_emotion, intensity_level, context)
        templates = self.empathy_templates.get(template_type, self.empathy_templates["validation"])
        
        # Choose specific template
        import random
        template = random.choice(templates)
        
        # Fill template with context
        response_text = self._fill_empathy_template(template, dominant_emotion, intensity_level, context)
        
        # Generate additional supportive elements
        supportive_elements = self._generate_supportive_elements(emotion_vector, context)
        
        return {
            "empathetic_response": response_text,
            "dominant_emotion": dominant_emotion,
            "intensity_level": intensity_level.value[2],
            "supportive_elements": supportive_elements,
            "emotional_complexity": emotion_vector.emotional_complexity(),
            "recommended_approach": self._recommend_approach(emotion_vector, context)
        }
    
    def _categorize_intensity(self, intensity: float) -> EmotionIntensity:
        """Categorize emotion intensity"""
        for level in EmotionIntensity:
            min_val, max_val, _ = level.value
            if min_val <= intensity < max_val:
                return level
        return EmotionIntensity.VERY_HIGH
    
    def _select_empathy_template(self, emotion: str, intensity: EmotionIntensity, context: EmotionalContext = None) -> str:
        """Select appropriate empathy template type"""
        
        # High intensity emotions need validation first
        if intensity in [EmotionIntensity.HIGH, EmotionIntensity.VERY_HIGH]:
            return "validation"
        
        # Negative emotions benefit from normalization
        if emotion in ["sadness", "fear", "anger", "disgust"]:
            return "normalization"
        
        # Positive emotions can use support or gentle reframe
        if emotion in ["joy", "trust", "anticipation"]:
            return "support"
        
        # Default to validation
        return "validation"
    
    def _fill_empathy_template(self, template: str, emotion: str, intensity_level: EmotionIntensity, context: EmotionalContext = None) -> str:
        """Fill empathy template with specific details"""
        
        # Create emotion adjective
        emotion_adjective = f"{intensity_level.value[2]} {emotion}"
        
        # Determine situation type
        situation_type = "challenging situation"
        if context and context.situation:
            situation_type = context.situation
        
        # Fill template
        filled_template = template.format(
            emotion=emotion,
            emotion_adjective=emotion_adjective,
            situation_type=situation_type,
            subject="this situation",
            positive_quality="compassion"
        )
        
        return filled_template
    
    def _generate_supportive_elements(self, emotion_vector: EmotionVector, context: EmotionalContext = None) -> Dict[str, Any]:
        """Generate additional supportive elements"""
        
        dominant_emotion, intensity = emotion_vector.dominant_emotion()
        
        elements = {
            "validation_phrases": [],
            "coping_suggestions": [],
            "reframe_opportunities": [],
            "check_in_prompts": []
        }
        
        # Validation phrases
        if intensity > 0.6:
            elements["validation_phrases"] = [
                "Your feelings are completely understandable",
                "It takes strength to acknowledge these emotions",
                "Thank you for trusting me with how you're feeling"
            ]
        
        # Coping suggestions based on emotion
        if dominant_emotion == "anxiety" or dominant_emotion == "fear":
            elements["coping_suggestions"] = [
                "Try some deep breathing exercises",
                "Ground yourself by naming 5 things you can see",
                "Remember that this feeling will pass"
            ]
        elif dominant_emotion == "sadness":
            elements["coping_suggestions"] = [
                "Allow yourself to feel this sadness",
                "Reach out to someone you trust",
                "Be gentle with yourself right now"
            ]
        elif dominant_emotion == "anger":
            elements["coping_suggestions"] = [
                "Take some time to cool down",
                "Express your feelings in a healthy way",
                "Consider what's really behind this anger"
            ]
        
        # Check-in prompts
        elements["check_in_prompts"] = [
            "How are you taking care of yourself right now?",
            "What would be most helpful for you in this moment?",
            "Is there anything specific you need support with?"
        ]
        
        return elements
    
    def _recommend_approach(self, emotion_vector: EmotionVector, context: EmotionalContext = None) -> Dict[str, Any]:
        """Recommend approach for responding to emotional state"""
        
        dominant_emotion, intensity = emotion_vector.dominant_emotion()
        complexity = emotion_vector.emotional_complexity()
        
        approach = {
            "primary_strategy": "validation",
            "tone": "warm and supportive",
            "pacing": "gentle",
            "focus_areas": [],
            "avoid": []
        }
        
        # Adjust based on intensity
        if intensity > 0.8:
            approach["primary_strategy"] = "immediate_support"
            approach["pacing"] = "slow and careful"
            approach["avoid"] = ["advice", "solutions", "minimizing"]
        elif intensity < 0.3:
            approach["primary_strategy"] = "gentle_exploration"
            approach["pacing"] = "normal"
        
        # Adjust based on complexity
        if complexity > 0.5:
            approach["focus_areas"] = ["acknowledge_complexity", "help_sort_feelings"]
        else:
            approach["focus_areas"] = ["focus_on_primary_emotion"]
        
        # Emotion-specific adjustments
        if dominant_emotion in ["anger", "frustration"]:
            approach["tone"] = "calm and understanding"
            approach["avoid"].extend(["confrontation", "judgment"])
        elif dominant_emotion in ["sadness", "grief"]:
            approach["tone"] = "gentle and compassionate"
            approach["focus_areas"].append("allow_processing_time")
        elif dominant_emotion in ["fear", "anxiety"]:
            approach["tone"] = "reassuring and stable"
            approach["focus_areas"].append("provide_grounding")
        
        return approach
    
    def track_emotional_journey(self, emotion_vector: EmotionVector, user_input: str, agent_response: str, context: EmotionalContext = None) -> EmotionalMemory:
        """Track emotional journey and learn from interactions"""
        
        # Create emotional memory
        memory = EmotionalMemory(
            timestamp=datetime.now(),
            emotion_vector=emotion_vector,
            context=context or EmotionalContext([], "", "", "", [], []),
            user_response=user_input,
            agent_response=agent_response,
            effectiveness_score=0.5,  # Will be updated based on feedback
            learned_patterns=[]
        )
        
        # Store in database
        self._store_emotional_memory(memory)
        
        # Add to in-memory cache
        self.emotional_memory.append(memory)
        
        # Keep only recent memories in cache
        if len(self.emotional_memory) > 100:
            self.emotional_memory = self.emotional_memory[-100:]
        
        return memory
    
    def _store_emotional_memory(self, memory: EmotionalMemory):
        """Store emotional memory in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO emotional_memories 
            (timestamp, emotion_vector, context, user_response, agent_response, effectiveness_score, learned_patterns)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory.timestamp.isoformat(),
            json.dumps(asdict(memory.emotion_vector)),
            json.dumps(asdict(memory.context)),
            memory.user_response,
            memory.agent_response,
            memory.effectiveness_score,
            json.dumps(memory.learned_patterns)
        ))
        
        conn.commit()
        conn.close()
    
    def analyze_emotional_patterns(self, days: int = 7) -> Dict[str, Any]:
        """Analyze emotional patterns over time"""
        
        # Get recent memories
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_memories = [m for m in self.emotional_memory if m.timestamp > cutoff_date]
        
        if not recent_memories:
            return {"message": "No recent emotional data available"}
        
        # Analyze patterns
        emotion_trends = self._analyze_emotion_trends(recent_memories)
        trigger_patterns = self._analyze_trigger_patterns(recent_memories)
        effectiveness_trends = self._analyze_effectiveness_trends(recent_memories)
        
        return {
            "analysis_period": f"{days} days",
            "total_interactions": len(recent_memories),
            "emotion_trends": emotion_trends,
            "trigger_patterns": trigger_patterns,
            "effectiveness_trends": effectiveness_trends,
            "recommendations": self._generate_emotional_recommendations(recent_memories)
        }
    
    def _analyze_emotion_trends(self, memories: List[EmotionalMemory]) -> Dict[str, Any]:
        """Analyze trends in emotional states"""
        
        emotion_totals = {emotion: 0.0 for emotion in EmotionVector.__annotations__.keys()}
        emotion_counts = {emotion: 0 for emotion in EmotionVector.__annotations__.keys()}
        
        for memory in memories:
            emotion_dict = asdict(memory.emotion_vector)
            for emotion, value in emotion_dict.items():
                if value > 0.1:  # Only count significant emotions
                    emotion_totals[emotion] += value
                    emotion_counts[emotion] += 1
        
        # Calculate averages
        emotion_averages = {}
        for emotion in emotion_totals:
            if emotion_counts[emotion] > 0:
                emotion_averages[emotion] = emotion_totals[emotion] / emotion_counts[emotion]
            else:
                emotion_averages[emotion] = 0.0
        
        # Find dominant emotions
        dominant_emotions = sorted(emotion_averages.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "dominant_emotions": dominant_emotions,
            "emotion_averages": emotion_averages,
            "emotional_diversity": len([e for e in emotion_averages.values() if e > 0.1])
        }
    
    def _analyze_trigger_patterns(self, memories: List[EmotionalMemory]) -> Dict[str, Any]:
        """Analyze patterns in emotional triggers"""
        
        trigger_counts = {}
        trigger_emotions = {}
        
        for memory in memories:
            for trigger in memory.context.triggers:
                if trigger not in trigger_counts:
                    trigger_counts[trigger] = 0
                    trigger_emotions[trigger] = []
                
                trigger_counts[trigger] += 1
                dominant_emotion, _ = memory.emotion_vector.dominant_emotion()
                trigger_emotions[trigger].append(dominant_emotion)
        
        # Find most common triggers
        common_triggers = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "common_triggers": common_triggers,
            "trigger_emotion_patterns": {
                trigger: max(set(emotions), key=emotions.count) 
                for trigger, emotions in trigger_emotions.items() if emotions
            }
        }
    
    def _analyze_effectiveness_trends(self, memories: List[EmotionalMemory]) -> Dict[str, Any]:
        """Analyze effectiveness of emotional responses"""
        
        if not memories:
            return {"average_effectiveness": 0.5}
        
        effectiveness_scores = [m.effectiveness_score for m in memories]
        average_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
        
        # Analyze by emotion type
        emotion_effectiveness = {}
        for memory in memories:
            dominant_emotion, _ = memory.emotion_vector.dominant_emotion()
            if dominant_emotion not in emotion_effectiveness:
                emotion_effectiveness[dominant_emotion] = []
            emotion_effectiveness[dominant_emotion].append(memory.effectiveness_score)
        
        # Calculate averages by emotion
        emotion_avg_effectiveness = {}
        for emotion, scores in emotion_effectiveness.items():
            emotion_avg_effectiveness[emotion] = sum(scores) / len(scores)
        
        return {
            "average_effectiveness": average_effectiveness,
            "emotion_effectiveness": emotion_avg_effectiveness,
            "improvement_trend": self._calculate_improvement_trend(effectiveness_scores)
        }
    
    def _calculate_improvement_trend(self, scores: List[float]) -> str:
        """Calculate if effectiveness is improving over time"""
        if len(scores) < 3:
            return "insufficient_data"
        
        # Compare first half to second half
        mid_point = len(scores) // 2
        first_half_avg = sum(scores[:mid_point]) / mid_point
        second_half_avg = sum(scores[mid_point:]) / (len(scores) - mid_point)
        
        if second_half_avg > first_half_avg + 0.1:
            return "improving"
        elif second_half_avg < first_half_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _generate_emotional_recommendations(self, memories: List[EmotionalMemory]) -> List[str]:
        """Generate recommendations based on emotional patterns"""
        
        recommendations = []
        
        # Analyze recent memories
        if not memories:
            return ["Continue building emotional awareness through regular check-ins"]
        
        # Check for high-stress patterns
        high_stress_count = sum(1 for m in memories if m.emotion_vector.magnitude() > 0.7)
        if high_stress_count > len(memories) * 0.3:
            recommendations.append("Consider stress management techniques - high emotional intensity detected")
        
        # Check for emotional diversity
        emotion_dict = {}
        for memory in memories:
            dominant_emotion, _ = memory.emotion_vector.dominant_emotion()
            emotion_dict[dominant_emotion] = emotion_dict.get(dominant_emotion, 0) + 1
        
        if len(emotion_dict) < 3:
            recommendations.append("Explore a wider range of emotional experiences for better emotional intelligence")
        
        # Check effectiveness trends
        recent_effectiveness = [m.effectiveness_score for m in memories[-5:]]
        if recent_effectiveness and sum(recent_effectiveness) / len(recent_effectiveness) < 0.4:
            recommendations.append("Focus on developing more effective emotional coping strategies")
        
        return recommendations

# Example usage
if __name__ == "__main__":
    ei = AdvancedEmotionalIntelligence()
    
    # Test emotional analysis
    text = "I'm feeling really anxious about the presentation tomorrow. I'm worried I'll mess up."
    context = {"situation": "work_presentation", "social_context": "professional"}
    
    emotion_vector = ei.analyze_emotional_content(text, context)
    empathetic_response = ei.generate_empathetic_response(emotion_vector)
    
    print("=== Emotional Analysis ===")
    print(f"Dominant emotion: {emotion_vector.dominant_emotion()}")
    print(f"Emotional complexity: {emotion_vector.emotional_complexity():.2f}")
    print(f"Empathetic response: {empathetic_response['empathetic_response']}")