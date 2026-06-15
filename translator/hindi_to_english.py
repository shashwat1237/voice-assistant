from huggingface_hub import InferenceClient
import os
from orchestration.error_handlers import LLMTimeoutError

client = InferenceClient(token=os.environ.get("HF_TOKEN"))
PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "translator_prompt.txt")

def translate_hi_to_en(hindi_text: str) -> str:
    with open(PROMPT_PATH, "r") as f:
        system_prompt = f.read()
    
    prompt = f"{system_prompt}\n\nHindi: {hindi_text}\nEnglish:"
    
    try:
        response = client.text_generation(
            prompt, 
            model="google/gemma-2b-it", 
            max_new_tokens=50
        )
        return response.strip()
    except Exception:
        raise LLMTimeoutError()
