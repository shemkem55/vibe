"""
Semantic Memory - Phase 1.1 Intelligence Upgrade
Enhanced memory with semantic search and user fact extraction
"""

import sqlite3
import json
import re
import math
from collections import Counter, defaultdict
from datetime import datetime


class SemanticMemory:
    """
    Enhanced memory layer with semantic similarity search
    Uses TF-IDF for lightweight vector-like similarity (no ML dependencies)
    """
    
    def __init__(self, db_path='agent_memory.db'):
        self.db_path = db_path
        self._init_semantic_tables()
        
        # In-memory index for fast search
        self.document_vectors = {}  # doc_id -> term frequencies
        self.idf_cache = {}  # term -> idf score
        self.all_terms = set()
        
        # Load existing memories into index
        self._build_index()
    
    def _init_semantic_tables(self):
        """Initialize semantic-specific tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # User facts (things learned about the user)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_type TEXT,
                    fact_subject TEXT,
                    fact_value TEXT,
                    confidence REAL DEFAULT 0.5,
                    source_text TEXT,
                    times_referenced INTEGER DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_referenced DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Semantic embeddings (simplified - stores term vectors as JSON)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER,
                    memory_type TEXT,
                    terms_json TEXT,
                    content_preview TEXT,
                    importance_score REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Conversation summaries (for long-term recall)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    main_topics TEXT,
                    key_points TEXT,
                    emotional_tone TEXT,
                    user_mentions TEXT,
                    exchange_count INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def _build_index(self):
        """Build in-memory search index from existing memories"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Load embeddings
            cursor.execute('SELECT id, terms_json, content_preview FROM memory_embeddings')
            for row in cursor.fetchall():
                try:
                    terms = json.loads(row['terms_json'])
                    self.document_vectors[row['id']] = {
                        "terms": terms,
                        "preview": row['content_preview']
                    }
                    self.all_terms.update(terms.keys())
                except json.JSONDecodeError:
                    continue
            
            # Calculate IDF for all terms
            self._calculate_idf()
    
    def _calculate_idf(self):
        """Calculate inverse document frequency for all terms"""
        total_docs = len(self.document_vectors) or 1
        term_doc_counts = Counter()
        
        for doc in self.document_vectors.values():
            for term in doc["terms"].keys():
                term_doc_counts[term] += 1
        
        for term, count in term_doc_counts.items():
            self.idf_cache[term] = math.log(total_docs / (1 + count)) + 1
    
    def _text_to_terms(self, text):
        """Convert text to term frequency dictionary"""
        # Tokenize and normalize
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        
        # Remove stopwords
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "and", "but", "if",
            "or", "because", "until", "while", "although", "i", "me", "my", "you",
            "your", "he", "him", "his", "she", "her", "it", "its", "we", "they",
            "them", "their", "what", "which", "who", "whom", "this", "that", "these",
            "those", "am", "about", "like", "really", "think", "know", "want", "tell"
        }
        
        filtered = [w for w in words if w not in stopwords and len(w) > 2]
        return dict(Counter(filtered))
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two term vectors"""
        # Get all terms
        all_terms = set(vec1.keys()) | set(vec2.keys())
        
        dot_product = 0
        norm1 = 0
        norm2 = 0
        
        for term in all_terms:
            tf1 = vec1.get(term, 0)
            tf2 = vec2.get(term, 0)
            idf = self.idf_cache.get(term, 1)
            
            tfidf1 = tf1 * idf
            tfidf2 = tf2 * idf
            
            dot_product += tfidf1 * tfidf2
            norm1 += tfidf1 ** 2
            norm2 += tfidf2 ** 2
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))
    
    def store_memory(self, content, memory_type="conversation", importance=0.5):
        """Store a memory with semantic embedding"""
        terms = self._text_to_terms(content)
        preview = content[:200] if len(content) > 200 else content
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO memory_embeddings 
                (memory_type, terms_json, content_preview, importance_score)
                VALUES (?, ?, ?, ?)
            ''', (memory_type, json.dumps(terms), preview, importance))
            memory_id = cursor.lastrowid
        
        # Update in-memory index
        self.document_vectors[memory_id] = {"terms": terms, "preview": preview}
        self.all_terms.update(terms.keys())
        self._calculate_idf()
        
        return memory_id
    
    def semantic_search(self, query, top_k=5, min_similarity=0.1):
        """Find memories semantically similar to query"""
        query_terms = self._text_to_terms(query)
        
        if not query_terms:
            return []
        
        # Calculate similarity to all documents
        similarities = []
        for doc_id, doc_data in self.document_vectors.items():
            sim = self._cosine_similarity(query_terms, doc_data["terms"])
            if sim >= min_similarity:
                similarities.append((doc_id, sim, doc_data["preview"]))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        results = []
        for doc_id, sim, preview in similarities[:top_k]:
            results.append({
                "memory_id": doc_id,
                "similarity": round(sim, 3),
                "content": preview
            })
            
            # Update access count
            self._record_access(doc_id)
        
        return results
    
    def _record_access(self, memory_id):
        """Record that a memory was accessed"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'UPDATE memory_embeddings SET access_count = access_count + 1 WHERE id = ?',
                (memory_id,)
            )
    
    # User Fact Extraction
    
    def extract_user_facts(self, user_text):
        """Extract facts about the user from their messages"""
        facts = []
        text_lower = user_text.lower()
        
        # Patterns for fact extraction
        patterns = {
            # Personal attributes
            r"(?:i am|i'm) (?:a |an )?(\w+(?:\s+\w+)?)": ("attribute", "identity"),
            r"my name is (\w+)": ("attribute", "name"),
            r"i (?:work|am working) (?:as |at |in )?(.+?)(?:\.|$|,)": ("attribute", "occupation"),
            
            # Preferences
            r"i (?:love|like|enjoy|prefer) (.+?)(?:\.|$|,)": ("preference", "likes"),
            r"i (?:hate|dislike|don't like|can't stand) (.+?)(?:\.|$|,)": ("preference", "dislikes"),
            r"my favorite (.+?) is (.+?)(?:\.|$|,)": ("preference", "favorite"),
            
            # Current state
            r"i (?:feel|am feeling) (\w+)": ("state", "feeling"),
            r"i(?:'m| am) (?:currently )?(\w+ing)": ("state", "activity"),
            
            # Interests
            r"i(?:'m| am) interested in (.+?)(?:\.|$|,)": ("interest", "topic"),
            r"i(?:'ve| have) been (?:learning|studying|researching) (.+?)(?:\.|$|,)": ("interest", "learning"),
            
            # Life events
            r"i (?:just|recently) (.+?)(?:\.|$|,)": ("event", "recent"),
            r"tomorrow i (?:will|am going to) (.+?)(?:\.|$|,)": ("event", "planned"),
        }
        
        for pattern, (fact_type, subject) in patterns.items():
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    value = " ".join(match)
                else:
                    value = match
                
                # Clean up the value
                value = value.strip().strip(",.!?")
                
                if value and len(value) > 2:
                    facts.append({
                        "type": fact_type,
                        "subject": subject,
                        "value": value,
                        "source": user_text[:100]
                    })
        
        return facts
    
    def store_user_fact(self, fact_type, subject, value, source_text="", confidence=0.7):
        """Store a fact about the user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if fact already exists
            cursor.execute('''
                SELECT id, times_referenced FROM user_facts 
                WHERE fact_type = ? AND fact_subject = ? AND fact_value = ?
            ''', (fact_type, subject, value))
            existing = cursor.fetchone()
            
            if existing:
                # Update reference count and timestamp
                conn.execute('''
                    UPDATE user_facts 
                    SET times_referenced = times_referenced + 1,
                        last_referenced = ?,
                        confidence = MIN(1.0, confidence + 0.1)
                    WHERE id = ?
                ''', (datetime.now(), existing[0]))
            else:
                # Insert new fact
                conn.execute('''
                    INSERT INTO user_facts 
                    (fact_type, fact_subject, fact_value, confidence, source_text)
                    VALUES (?, ?, ?, ?, ?)
                ''', (fact_type, subject, value, confidence, source_text[:200]))
    
    def get_user_facts(self, fact_type=None, min_confidence=0.5):
        """Retrieve stored facts about the user"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if fact_type:
                cursor.execute('''
                    SELECT * FROM user_facts 
                    WHERE fact_type = ? AND confidence >= ?
                    ORDER BY times_referenced DESC, confidence DESC
                ''', (fact_type, min_confidence))
            else:
                cursor.execute('''
                    SELECT * FROM user_facts 
                    WHERE confidence >= ?
                    ORDER BY times_referenced DESC, confidence DESC
                ''', (min_confidence,))
            
            return [dict(r) for r in cursor.fetchall()]
    
    def get_user_profile(self):
        """Get a summary profile of the user based on stored facts"""
        all_facts = self.get_user_facts(min_confidence=0.4)
        
        profile = {
            "identity": [],
            "preferences": {"likes": [], "dislikes": [], "favorites": []},
            "interests": [],
            "recent_states": [],
            "total_facts": len(all_facts)
        }
        
        for fact in all_facts:
            ft = fact["fact_type"]
            fs = fact["fact_subject"]
            fv = fact["fact_value"]
            
            if ft == "attribute":
                profile["identity"].append(fv)
            elif ft == "preference":
                if fs == "likes":
                    profile["preferences"]["likes"].append(fv)
                elif fs == "dislikes":
                    profile["preferences"]["dislikes"].append(fv)
                elif fs == "favorite":
                    profile["preferences"]["favorites"].append(fv)
            elif ft == "interest":
                profile["interests"].append(fv)
            elif ft == "state":
                profile["recent_states"].append(fv)
        
        return profile
    
    # Conversation Summarization
    
    def summarize_conversation(self, exchanges, date=None):
        """Create a summary of a conversation for long-term storage"""
        if not exchanges:
            return None
        
        date = date or datetime.now().strftime("%Y-%m-%d")
        
        # Extract main topics
        all_text = " ".join([ex.get("user", "") + " " + ex.get("agent", "") for ex in exchanges])
        terms = self._text_to_terms(all_text)
        main_topics = sorted(terms.items(), key=lambda x: x[1], reverse=True)[:5]
        topic_str = ", ".join([t[0] for t in main_topics])
        
        # Extract key points (sentences with questions or statements)
        key_points = []
        for ex in exchanges[:10]:  # Limit to first 10 for key points
            user_msg = ex.get("user", "")
            if "?" in user_msg and len(user_msg) > 10:
                key_points.append(f"Q: {user_msg[:100]}")
        
        # Detect emotional tone (simplified)
        emotional_markers = {"positive": 0, "negative": 0, "curious": 0, "neutral": 0}
        positive_words = ["love", "great", "amazing", "happy", "thanks", "wonderful"]
        negative_words = ["sad", "hate", "bad", "terrible", "wrong", "frustrated"]
        curious_words = ["why", "how", "what", "wonder", "curious"]
        
        for word in all_text.lower().split():
            if word in positive_words:
                emotional_markers["positive"] += 1
            elif word in negative_words:
                emotional_markers["negative"] += 1
            elif word in curious_words:
                emotional_markers["curious"] += 1
        
        dominant_tone = max(emotional_markers, key=emotional_markers.get)
        
        # Store summary
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO conversation_summaries 
                (date, main_topics, key_points, emotional_tone, exchange_count)
                VALUES (?, ?, ?, ?, ?)
            ''', (date, topic_str, json.dumps(key_points), dominant_tone, len(exchanges)))
        
        return {
            "date": date,
            "topics": topic_str,
            "key_points": key_points,
            "tone": dominant_tone,
            "exchanges": len(exchanges)
        }
    
    def recall_past_conversations(self, query=None, days_back=30):
        """Recall summaries of past conversations"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if query:
                # Search by topic
                cursor.execute('''
                    SELECT * FROM conversation_summaries 
                    WHERE main_topics LIKE ?
                    ORDER BY created_at DESC LIMIT 10
                ''', (f"%{query.lower()}%",))
            else:
                # Get recent
                cursor.execute('''
                    SELECT * FROM conversation_summaries 
                    ORDER BY created_at DESC LIMIT 10
                ''')
            
            return [dict(r) for r in cursor.fetchall()]


if __name__ == "__main__":
    # Test semantic memory
    mem = SemanticMemory()
    
    print("=== Semantic Memory Test ===\n")
    
    # Store some memories
    mem.store_memory("We talked about quantum physics and the nature of reality", "conversation", 0.8)
    mem.store_memory("User asked about their career path in software engineering", "conversation", 0.7)
    mem.store_memory("Discussion about the meaning of life and consciousness", "conversation", 0.9)
    mem.store_memory("User mentioned they love playing guitar and music", "conversation", 0.6)
    
    # Test semantic search
    print("Searching for 'physics and science'...")
    results = mem.semantic_search("physics and science")
    for r in results:
        print(f"  [{r['similarity']:.2f}] {r['content'][:60]}...")
    
    # Test user fact extraction
    print("\n--- User Fact Extraction ---")
    test_inputs = [
        "I'm a software developer working at a startup",
        "I love hiking and photography",
        "My name is Alex and I'm learning machine learning",
        "I feel excited about the future of AI"
    ]
    
    for text in test_inputs:
        print(f"\nInput: '{text}'")
        facts = mem.extract_user_facts(text)
        for f in facts:
            print(f"  -> {f['type']}/{f['subject']}: {f['value']}")
            mem.store_user_fact(f['type'], f['subject'], f['value'], text)
    
    # Get user profile
    print("\n--- User Profile ---")
    profile = mem.get_user_profile()
    for key, value in profile.items():
        if value:
            print(f"  {key}: {value}")
