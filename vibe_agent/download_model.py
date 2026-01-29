"""
Model Downloader
Downloads the requested model for local use.
"""

import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def download_model(model_id="microsoft/Phi-3-mini-4k-instruct", save_dir="./models/phi-3-mini"):
    print(f"🚀 Starting download for {model_id}...")
    print(f"📂 Destination: {save_dir}")
    
    if os.path.exists(save_dir):
        print("⚠️ Model directory already exists. Checking contents...")
        if os.path.exists(os.path.join(save_dir, "config.json")):
            print("✅ Model appears to be present. Use --force to redownload.")
            return

    try:
        print("⏳ Downloading Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.save_pretrained(save_dir)
        
        print("⏳ Downloading Model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        model.save_pretrained(save_dir)
        
        print("🎉 Download complete!")
        print(f"Path: {os.path.abspath(save_dir)}")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")

if __name__ == "__main__":
    download_model()
