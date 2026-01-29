import sqlite3
import json
import os
import time

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_dashboard():
    db_path = 'agent_memory.db'
    essence_path = 'essence.json'
    
    while True:
        clear_screen()
        print("🌌 Vibe-Code Agent Dashboard | Phase 6.2 Sanctuary\n")
        
        # 1. Essence Summary
        if os.path.exists(essence_path):
            with open(essence_path, 'r') as f:
                ess = json.load(f)
                print(f"[Identity]: {ess.get('name')} | Rhythm: {ess['speech_traits']['rhythm']}")
                print(f"[Focus]: {', '.join(ess['knowledge_mood']['primary_interests'])}")
        
        # 2. Memory Stats
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM session_log")
            total_turns = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM episodic_memory")
            memories = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM golden_responses")
            blessed = cursor.fetchone()[0]
            
            print(f"\n[Mind Stats]: {total_turns} interactions | {memories} episodic memories | {blessed} blessed responses")
            
            # 3. Recent Activity
            print("\n[Recent Consciousness Flow]:")
            cursor.execute("SELECT role, content, timestamp FROM session_log ORDER BY id DESC LIMIT 5")
            logs = cursor.fetchall()
            for role, content, ts in reversed(logs):
                color = "\033[94m" if role == "user" else "\033[92m"
                reset = "\033[0m"
                trunc_content = content[:60] + "..." if len(content) > 60 else content
                print(f"  {ts[11:19]} {color}{role.upper():>5}{reset}: {trunc_content}")
            
            conn.close()
        
        print("\n" + "-"*60)
        print("Refreshing in 10 seconds... (Ctrl+C to exit)")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("\nLeaving the sanctuary.")
            break

if __name__ == "__main__":
    show_dashboard()
