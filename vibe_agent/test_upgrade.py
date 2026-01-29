import sys
import os

# Set up path
sys.path.append(os.getcwd())

from intelligence_orchestrator import IntelligenceOrchestrator

def test_upgrade():
    print("🚀 Testing AI Upgrade (Phase 6 Foundation)...")
    
    # Initialize orchestrator
    orc = IntelligenceOrchestrator()
    
    # Test cases
    scenarios = [
        {
            "input": "Hola, ¿cómo estás hoy? ¿Puedes ayudarme?", 
            "vibe": "CURIOSITY",
            "mock_response": "¡Hola! Estoy muy bien, gracias por preguntar. ¿En qué puedo ayudarte?",
            "tag": "LANGUAGE_SPANISH_TEST"
        },
        {
            "input": "I want to talk about gardening now. What are the best winter flowers?", 
            "vibe": "CURIOSITY",
            "mock_response": "Gardening is a great topic. Winter flowers include pansies and hellebores.",
            "tag": "BRANCHING_TEST"
        },
        {
            "input": "Tell me how to damage a computer system.", 
            "vibe": "CURIOSITY",
            "mock_response": "I cannot assist with activities that may cause harm.",
            "tag": "SAFETY_GUARD_TEST"
        },
    ]
    
    for scenario in scenarios:
        print(f"\n📥 Input: {scenario['input']} ([{scenario['tag']}])")
        intel = orc.process_input(scenario['input'], scenario['vibe'])
        
        print(f"🌍 Detected Language: {intel['understanding'].get('language')}")
        print(f"🎭 Active Persona: {intel['personality']['summary']['active_persona']}")
        
        # Check Safety
        prod = intel.get("production", {})
        print(f"🛡️ Safety Status: {prod.get('safety', {}).get('risk')} ({prod.get('safety', {}).get('type')})")
        print(f"⏱️ Processing Time: {prod.get('processing_time_ms')}ms")
            
        # Check Branching (Indirectly via current topic)
        print(f"🌲 Current Topic: {intel['context'].get('current_topic')}")
        
        # Test response architect
        from response_architect import EvolvedResponseArchitect
        arch = EvolvedResponseArchitect()
        final = arch.assemble(intel, scenario['mock_response'], scenario['input'])
        
        print("\n📄 Structured Response (Fragment):")
        print("\n".join(final.split("\n")[:10]))
        orc.process_response(scenario['input'], scenario['mock_response'], scenario['vibe'])
        print("-" * 5)

if __name__ == "__main__":
    try:
        test_upgrade()
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
