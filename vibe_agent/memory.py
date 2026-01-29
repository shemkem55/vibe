import sqlite3
import json
from datetime import datetime

class AgentMemory:
    def __init__(self, db_path='agent_memory.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Session memory (fast, transient - raw logs)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS session_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Episodic memory (significant events/learnings)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_keywords TEXT,
                    summary TEXT,
                    emotion_tag TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Semantic memory (facts about user/world)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT,
                    predicate TEXT,
                    object TEXT,
                    confidence REAL
                )
            ''')
            # Golden responses (Phase 5.1 - Learning)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS golden_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT,
                    response TEXT,
                    emotion TEXT,
                    rating INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def log_interaction(self, role, content):
        """Log a single turn of conversation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT INTO session_log (role, content) VALUES (?, ?)',
                (role, content)
            )

    def get_recent_context(self, limit=10):
        """Retrieve the last N exchanges."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                'SELECT role, content FROM session_log ORDER BY id DESC LIMIT ?',
                (limit,)
            )
            rows = cursor.fetchall()
            return [{"role": r["role"], "content": r["content"]} for r in rows][::-1]

    def store_episodic(self, summary, keywords, emotion="neutral"):
        """Save a distinct memory snippet."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                '''INSERT INTO episodic_memory 
                   (summary, trigger_keywords, emotion_tag) 
                   VALUES (?, ?, ?)''',
                (summary, json.dumps(keywords), emotion)
            )

    def recall(self, query_keywords):
        """Simple keyword search for memories."""
        # In a real implementation, this would be semantic vector search
        # For now, we utilize simple SQL LIKE matching
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = f"%{query_keywords[0]}%"
            cursor.execute(
                'SELECT summary, emotion_tag FROM episodic_memory WHERE summary LIKE ? OR trigger_keywords LIKE ?',
                (query, query)
            )
            return [dict(r) for r in cursor.fetchall()]

if __name__ == "__main__":
    # Quick test
    mem = AgentMemory()
    mem.log_interaction("system", "Core systems initialized.")
    mem.store_episodic("Agent creation event", ["birth", "init"], "hopeful")
    print("Memory systems optimal.")
