from huggingface_hub import InferenceClient
import os
from orchestration.error_handlers import LLMTimeoutError

client = InferenceClient(token=os.environ.get("HF_TOKEN"),timeout=120.0)
PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "translator_prompt.txt")

def translate_en_to_hi(english_text: str) -> str:
    with open(PROMPT_PATH, "r") as f:
        system_prompt = f.read()
    
    prompt = f"{system_prompt}\n\nEnglish: {english_text}\nHindi:"
    
    try:
        response = client.text_generation(
            prompt, 
            model="google/gemma-2b-it", 
            max_new_tokens=400
        )
        return response.strip()
    except Exception:
         raise LLMTimeoutError()
