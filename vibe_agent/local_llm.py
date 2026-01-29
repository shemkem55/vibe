"""
Local LLM Integration Module
Wraps local model inference with support for Phi-3 Mini and quantization.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LocalLLM:
    """Wrapper for local LLM inference with multi-profile support"""
    
    def __init__(self, model_path="./models/phi-3-mini", device="auto"):
        """
        Initialize local LLM
        Options for device: "cuda", "cpu", "auto"
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Expert Profiles (Simulating Multi-Model Orchestration)
        self.profiles = {
            "primary": {
                "temperature": 0.7,
                "top_p": 0.9,
                "system_prompt": "You are the primary reasoning core. Be balanced, accurate, and helpful."
            },
            "creative": {
                "temperature": 0.9,
                "top_p": 0.95,
                "system_prompt": "You are the creative synthesis engine. Be imaginative, use rich metaphors, and think outside the box."
            },
            "analytical": {
                "temperature": 0.3,
                "top_p": 0.8,
                "system_prompt": "You are the analytical processor. Be precise, logical, and focus on data and facts."
            },
            "technical": {
                "temperature": 0.2,
                "top_p": 0.7,
                "system_prompt": "You are the technical expert. Provide clear, efficient code and technical explanations."
            },
            "empathetic": {
                "temperature": 0.8,
                "top_p": 0.9,
                "system_prompt": "You are the emotional intelligence layer. Focus on empathy, validation, and human connection."
            }
        }
        
        # Generation parameters
        self.generation_config = {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.1,
        }
        
    def load_model(self, quantization=None):
        """
        Lazy loading of model
        quantization: '4bit', '8bit', or None
        """
        if self.is_loaded:
            return True
        
        try:
            print(f"Loading LLM from {self.model_path}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            
            # Configure Quantization
            quant_config = None
            if quantization == "4bit":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            elif quantization == "8bit":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=self.device,
                quantization_config=quant_config,
                trust_remote_code=True
            )
            
            self.model.eval()
            self.is_loaded = True
            print(f"Model loaded successfully! (Quantization: {quantization})")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def generate(self, prompt, profile="primary", **kwargs):
        """Generate text from prompt using specific profile"""
        if not self.is_loaded:
            success = self.load_model(quantization="4bit")
            if not success:
                return "Error: Model failed to load"
        
        try:
            # Apply profile parameters
            profile_cfg = self.profiles.get(profile, self.profiles["primary"])
            gen_config = self.generation_config.copy()
            
            # Apply dynamic adjustments (Expansion)
            temp_adj = kwargs.pop("temperature_adj", 0.0)
            top_p_adj = kwargs.pop("top_p_adj", 0.0)
            
            gen_config.update({
                "temperature": max(0.1, min(2.0, profile_cfg["temperature"] + temp_adj)),
                "top_p": max(0.1, min(1.0, profile_cfg["top_p"] + top_p_adj))
            })
            gen_config.update(kwargs)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            )
            
            if hasattr(self.model, "device"):
                inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_config
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return generated_text
            
        except Exception as e:
            return f"Error during generation: {str(e)}"
    
    def format_chat_prompt(self, messages, profile="primary", custom_system=None):
        """Format messages for Phi-3 with profile-based system prompt"""
        profile_cfg = self.profiles.get(profile, self.profiles["primary"])
        sys_prompt = custom_system or profile_cfg["system_prompt"]
        
        formatted = [f"<|user|>\n{sys_prompt}<|end|>\n<|assistant|>\nUnderstood.<|end|>\n"]
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                formatted.append(f"<|user|>\n{content}<|end|>\n")
            elif role == "assistant":
                formatted.append(f"<|assistant|>\n{content}<|end|>\n")
            
        formatted.append("<|assistant|>")
        return "".join(formatted)

    def chat(self, messages, profile="primary", system_prompt=None, **kwargs):
        """Chat-style interaction with profile support and adaptation"""
        prompt = self.format_chat_prompt(messages, profile, system_prompt)
        return self.generate(prompt, profile=profile, **kwargs)

# Singleton instance for easy import
_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LocalLLM(model_path="./models/phi-3-mini")
    return _llm_instance
