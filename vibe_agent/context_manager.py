"""
Intelligent Context Manager - Phase 1 Intelligence Upgrade
Tracks conversation flow, extracts topics, and maintains working memory
"""

from collections import Counter
import re
from datetime import datetime

class ContextManager:
    """Maintain conversation context beyond just last message"""
    
    def __init__(self, window_size=10):
        self.context_window = []
        self.max_window = window_size
        self.context_summary = ""
        self.current_topic = None
        self.user_interests = set()
        self.last_questions = []
        
        # 🆕 Branching Conversation Support (Expansion)
        self.threads = {"main": []} # thread_id -> [exchanges]
        self.active_thread = "main"
        self.thematic_branches = {} # topic -> thread_id
        
    def switch_thread(self, thread_id: str):
        """Switch the active conversation branch"""
        if thread_id not in self.threads:
            self.threads[thread_id] = []
        self.active_thread = thread_id
        # Update main window for compatibility
        self.context_window = self.threads[thread_id]

    def create_branch(self, topic: str):
        """Create a new conversation branch for a specific topic"""
        thread_id = f"thread_{len(self.threads)}"
        self.threads[thread_id] = []
        self.thematic_branches[topic.lower()] = thread_id
        self.switch_thread(thread_id)
        return thread_id
        
    def update_context(self, user_input, agent_response, vibe):
        """Update and summarize context"""
        topics = self._extract_keywords(user_input)
        
        # 🆕 Branching Logic: If new topic detected and not related to current, consider branching
        if self.current_topic and topics and self.current_topic not in topics:
            # Topic shift detected
            new_topic = topics[0]
            if new_topic not in self.thematic_branches:
                print(f"🌲 New thematic branch detected: {new_topic}")
                # self.create_branch(new_topic) # Auto-branching could be toggled
        
        entry = {
            "user": user_input,
            "agent": agent_response,
            "vibe": vibe,
            "time": datetime.now(),
            "topics": topics
        }
        
        # Append to active thread and main window
        self.threads[self.active_thread].append(entry)
        self.context_window = self.threads[self.active_thread]
        
        # Track topics
        for topic in entry["topics"]:
            self.user_interests.add(topic)
        
        # Track questions
        if "?" in user_input:
            self.last_questions.append(user_input)
            if len(self.last_questions) > 5:
                self.last_questions.pop(0)
        
        # Maintain window size
        if len(self.context_window) > self.max_window:
            self.context_window.pop(0)
        
        # Update summary and topic on every interaction (Upgraded for Real-time)
        self.context_summary = self._summarize_context()
        self.current_topic = self._identify_current_topic()
    
    def _extract_keywords(self, text):
        """Extract important keywords from text"""
        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", 
                    "to", "for", "of", "with", "from", "that", "this", "it", "as",
                    "i", "you", "me", "my", "your", "and", "or", "but"}
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        
        return keywords[:5]  # Top 5 keywords
    
    def _identify_current_topic(self):
        """Identify the main topic of recent conversation"""
        if not self.context_window:
            return None
        
        # Get keywords from last 3 exchanges
        all_keywords = []
        for exchange in self.context_window[-3:]:
            all_keywords.extend(exchange.get("topics", []))
        
        if not all_keywords:
            return None
        
        # Find most common
        topic_counts = Counter(all_keywords)
        return topic_counts.most_common(1)[0][0] if topic_counts else None
    
    def _summarize_context(self):
        """Create intelligent summary of conversation so far"""
        if len(self.context_window) < 2:
            return ""
        
        topics = []
        for exchange in self.context_window[-5:]:
            topics.extend(exchange.get("topics", []))
        
        # Find most frequent topics
        topic_counts = Counter(topics)
        main_topics = [topic for topic, count in topic_counts.most_common(3)]
        
        # Build summary
        summary_parts = []
        if main_topics:
            summary_parts.append(f"Discussing: {', '.join(main_topics)}")
        
        # Note dominant vibe
        vibes = [ex["vibe"] for ex in self.context_window[-3:]]
        if vibes:
            dominant_vibe = Counter(vibes).most_common(1)[0][0]
            summary_parts.append(f"Mood: {dominant_vibe}")
        
        return ". ".join(summary_parts)
    
    def get_context_for_prompt(self):
        """Format context for intelligent response generation"""
        if not self.context_window:
            return None
        
        # Last 3 exchanges
        recent = self.context_window[-3:]
        
        context_data = {
            "recent_exchanges": [
                {"user": ex["user"], "agent": ex["agent"]} 
                for ex in recent
            ],
            "current_topic": self.current_topic,
            "summary": self.context_summary,
            "user_interests": list(self.user_interests)[-5:],  # Last 5 interests
            "conversation_length": len(self.context_window)
        }
        
        return context_data
    
    def is_follow_up_question(self, user_input):
        """Detect if this continues the previous topic"""
        if not self.current_topic or len(self.context_window) < 2:
            return False
        
        # Check if current input mentions current topic
        keywords = self._extract_keywords(user_input)
        return self.current_topic in keywords
    
    def is_repeating(self, user_input):
        """Check if user is repeating themselves"""
        if len(self.context_window) < 2:
            return False
        
        # Check last 3 user inputs
        recent_user_inputs = [ex["user"].lower() for ex in self.context_window[-3:]]
        
        # Simple similarity check
        user_lower = user_input.lower()
        for past_input in recent_user_inputs:
            # If 80% of words match, consider it a repetition
            past_words = set(past_input.split())
            current_words = set(user_lower.split())
            
            if past_words and current_words:
                overlap = len(past_words & current_words) / len(current_words)
                if overlap > 0.8:
                    return True
        
        return False
    
    def get_unanswered_questions(self):
        """Find questions that might not have been fully answered"""
        # Simple heuristic: questions followed by more questions
        unanswered = []
        
        for i, question in enumerate(self.last_questions[:-1]):
            # If next message was also a question, previous might be unanswered
            if i + 1 < len(self.last_questions):
                unanswered.append(question)
        
        return unanswered[-2:] if unanswered else []
