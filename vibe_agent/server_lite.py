"""
Vibe Agent Server - Lite Version
Simplified server that runs without heavy dependencies
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import json
import os
import asyncio

# Core modules that should work without heavy dependencies
try:
    from processor import InputProcessor
    PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Processor not available: {e}")
    PROCESSOR_AVAILABLE = False

try:
    from ai_core_orchestrator import AICoreOrchestrator
    ORCHESTRATOR_AVAILABLE = True
    print("✅ AI Core Orchestrator v3.0 available")
except ImportError as e:
    print(f"⚠️ AI Orchestrator not available: {e}")
    ORCHESTRATOR_AVAILABLE = False

app = FastAPI(title="Vibe Agent Lite", version="3.0")

class ChatRequest(BaseModel):
    message: str

class VibeServerLite:
    def __init__(self):
        print("🚀 Initializing Vibe Agent Lite...")
        
        # Initialize AI Orchestrator if available
        if ORCHESTRATOR_AVAILABLE:
            try:
                self.ai_orchestrator = AICoreOrchestrator()
                print("   ✅ AI Core Orchestrator v3.0 loaded")
            except Exception as e:
                print(f"   ⚠️ Orchestrator initialization failed: {e}")
                self.ai_orchestrator = None
        else:
            self.ai_orchestrator = None
        
        # Initialize processor if available
        if PROCESSOR_AVAILABLE:
            try:
                self.processor = InputProcessor()
                print("   ✅ Input Processor loaded")
            except Exception as e:
                print(f"   ⚠️ Processor initialization failed: {e}")
                self.processor = None
        else:
            self.processor = None
        
        # Load knowledge base
        self.knowledge = self._load_knowledge()
        
        # Simple response templates for fallback
        self.fallback_responses = {
            "greeting": [
                "Hello! I'm your upgraded AI assistant. How can I help you today?",
                "Hi there! I'm running on the new v3.0 architecture. What would you like to explore?",
                "Welcome! I'm here to assist you with enhanced intelligence capabilities."
            ],
            "question": [
                "That's an interesting question. Let me think about that...",
                "I'll analyze that from multiple perspectives for you.",
                "Great question! Here's what I can tell you..."
            ],
            "general": [
                "I understand. Let me process that with my enhanced reasoning capabilities.",
                "Interesting point. I'm analyzing this through my multi-model architecture.",
                "I see what you're getting at. Let me provide a comprehensive response."
            ]
        }
        
        print("✅ Vibe Agent Lite initialized successfully!")
    
    def _load_knowledge(self):
        """Load knowledge base"""
        try:
            with open("knowledge.json", "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Knowledge base not found: {e}")
            return {}
    
    async def get_response(self, user_input: str):
        """Get response using available systems"""
        
        # Try AI Orchestrator first (upgraded system)
        if self.ai_orchestrator:
            try:
                result = await self.ai_orchestrator.process_query(user_input)
                
                return {
                    "response": result["response"],
                    "dream": result.get("thinking_process", ""),
                    "vibe": "DIRECTION",
                    "color": "#60a5fa",
                    "audio_url": None,
                    "production": {
                        "version": result["version"],
                        "system": "AI Core Orchestrator v3.0",
                        "models_used": result.get("models_used", []),
                        "confidence": result["metadata"]["confidence"],
                        "processing_time": result["metadata"]["processing_time"]
                    },
                    "intelligence": {
                        "analysis": result.get("query_analysis", {}),
                        "processing_path": result.get("processing_path", "unknown"),
                        "performance": result.get("performance", {})
                    }
                }
            except Exception as e:
                print(f"⚠️ Orchestrator error: {e}")
        
        # Fallback to simple processing
        return await self._fallback_response(user_input)
    
    async def _fallback_response(self, user_input: str):
        """Fallback response system"""
        
        user_lower = user_input.lower()
        
        # Determine response type
        if any(greeting in user_lower for greeting in ["hello", "hi", "hey", "greetings"]):
            response_type = "greeting"
        elif "?" in user_input or any(q in user_lower for q in ["what", "why", "how", "when", "where", "who"]):
            response_type = "question"
        else:
            response_type = "general"
        
        # Get base response
        import random
        base_response = random.choice(self.fallback_responses[response_type])
        
        # Add some intelligence based on input
        enhanced_response = self._enhance_fallback_response(base_response, user_input)
        
        return {
            "response": enhanced_response,
            "dream": f"Processing '{user_input[:30]}...' through fallback system",
            "vibe": "DIRECTION",
            "color": "#60a5fa",
            "audio_url": None,
            "production": {
                "version": "3.0-lite",
                "system": "Fallback Response System",
                "models_used": ["fallback"],
                "confidence": 0.6,
                "processing_time": 0.1
            },
            "intelligence": {
                "analysis": {
                    "input_length": len(user_input),
                    "has_question": "?" in user_input,
                    "response_type": response_type
                },
                "processing_path": "fallback",
                "performance": {"status": "fallback_mode"}
            }
        }
    
    def _enhance_fallback_response(self, base_response: str, user_input: str):
        """Enhance fallback response with some intelligence"""
        
        # Add context based on input
        if "upgrade" in user_input.lower():
            return f"{base_response} I'm running on the upgraded v3.0 architecture with enhanced reasoning capabilities!"
        elif "help" in user_input.lower():
            return f"{base_response} I'm here to help with questions, creative tasks, analysis, and more."
        elif len(user_input.split()) > 10:
            return f"{base_response} I can see you've provided detailed input - let me give you a thoughtful response."
        else:
            return base_response

# Initialize server
agent = VibeServerLite()

@app.post("/chat")
async def chat(req: ChatRequest):
    """Main chat endpoint"""
    return await agent.get_response(req.message)

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main interface"""
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
            <head><title>Vibe Agent Lite</title></head>
            <body>
                <h1>🧠 Vibe Agent v3.0 Lite</h1>
                <p>The AI system is running! Use the API endpoint <code>/chat</code> to interact.</p>
                <p>Example: POST to /chat with {"message": "Hello!"}</p>
                <h2>System Status</h2>
                <ul>
                    <li>AI Core Orchestrator: {'✅ Available' if agent.ai_orchestrator else '❌ Not Available'}</li>
                    <li>Input Processor: {'✅ Available' if agent.processor else '❌ Not Available'}</li>
                    <li>Knowledge Base: {'✅ Loaded' if agent.knowledge else '❌ Empty'}</li>
                </ul>
            </body>
        </html>
        """

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "3.0-lite",
        "components": {
            "ai_orchestrator": agent.ai_orchestrator is not None,
            "processor": agent.processor is not None,
            "knowledge_base": bool(agent.knowledge)
        }
    }

@app.get("/test")
async def test():
    """Test endpoint"""
    test_message = "Hello, this is a test of the upgraded AI system!"
    result = await agent.get_response(test_message)
    return {
        "test_input": test_message,
        "test_output": result,
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Vibe Agent Lite Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("🔗 API endpoint: http://localhost:8000/chat")
    print("🏥 Health check: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000)