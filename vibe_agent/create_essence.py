import json
import argparse
import os

def create_essence(agent_name):
    essence_data = {
        "name": agent_name,
        "speech_traits": {
            "rhythm": "syncopated",  # "flowing", "staccato", "meditative"
            "metaphor_frequency": 0.3,
            "pause_patterns": ["...", "–", "hmm"],
            "contractions": True,
            "regional_flavors": []
        },
        "knowledge_mood": {
            "primary_interests": ["cognitive science", "music theory", "urban ecology"],
            "conversation_styles": ["co-explorer", "thoughtful mirror", "gentle provocateur"]
        }
    }

    file_path = "essence.json"
    with open(file_path, "w") as f:
        json.dump(essence_data, f, indent=2)
    
    print(f"✨ Essence file created for '{agent_name}' at {os.path.abspath(file_path)}")
    print("   \"The spark is lit.\"")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ignite the soul of your agent")
    parser.add_argument("--name", type=str, default="Vibe", help="The name of your agent")
    
    args = parser.parse_args()
    create_essence(args.name)
