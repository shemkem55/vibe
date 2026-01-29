import sqlite3
import json
import os
from datetime import datetime

class Gardener:
    def __init__(self, db_path='agent_memory.db', essence_path='essence.json'):
        self.db_path = db_path
        self.essence_path = essence_path

    def review_recent_chat(self, limit=10):
        """Display recent logs and allow the user to 'bless' them."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get last exchanges where the agent spoke
        cursor.execute('''
            SELECT u.content as user_msg, a.content as agent_msg, a.timestamp 
            FROM session_log u
            JOIN session_log a ON a.id = u.id + 1
            WHERE u.role = 'user' AND a.role = 'agent'
            ORDER BY a.id DESC LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            print("No recent exchanges found to review.")
            return

        print("\n--- 🌿 The Gardener: Reviewing Recent Growth ---")
        for i, row in enumerate(rows):
            print(f"\n[{i+1}] User: {row['user_msg']}")
            print(f"    {row['agent_msg']}")
        
        choice = input("\nEnter the ID of a response to 'Bless' (or 'q' to quit): ")
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(rows):
                self.bless_response(rows[idx]['user_msg'], rows[idx]['agent_msg'])

    def bless_response(self, prompt, response):
        """Save a high-quality response to the 'golden_responses' table."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO golden_responses (prompt, response, rating)
            VALUES (?, ?, 5)
        ''', (prompt, response))
        conn.commit()
        conn.close()
        print(f"✨ Response blessed and saved to Golden Examples.")

    def feed_knowledge(self, topic, description):
        """Manually add new knowledge seeds (Phase 5.1)."""
        knowledge_path = 'knowledge.json'
        if os.path.exists(knowledge_path):
            with open(knowledge_path, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        if topic not in data:
            data[topic] = []
        
        data[topic].append(description)
        
        with open(knowledge_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"📖 Seeded new knowledge: '{topic}'")

    def journal_reflection(self):
        """A space for the user to document the agent's evolution (Phase 5.1)."""
        reflection = input("\nWhat felt human today? What felt artificial? \n> ")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open("journal.md", "a") as f:
            f.write(f"\n## Reflection - {timestamp}\n")
            f.write(f"{reflection}\n")
        
        print("📝 Reflection saved to journal.md")

if __name__ == "__main__":
    gardener = Gardener()
    print("Welcome, Gardener.")
    print("1. Review/Bless recent chats")
    print("2. Feed new knowledge")
    print("3. Write journal reflection")
    
    cmd = input("\nSelect an action: ")
    if cmd == '1':
        gardener.review_recent_chat()
    elif cmd == '2':
        t = input("Topic: ")
        d = input("Description/Thought: ")
        gardener.feed_knowledge(t, d)
    elif cmd == '3':
        gardener.journal_reflection()
