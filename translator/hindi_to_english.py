from huggingface_hub import InferenceClient
from huggingface_hub.errors import InferenceTimeoutError, HfHubHTTPError
import os
from orchestration.error_handlers import LLMTimeoutError

# Timeout explicitly defined at initialization layer
client = InferenceClient(token=os.environ.get("HF_TOKEN"), timeout=15.0)
PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "translator_prompt.txt")

def translate_hi_to_en(hindi_text: str) -> str:
    with open(PROMPT_PATH, "r") as f:
        system_prompt = f.read()

    prompt = f"{system_prompt}\n\nHindi: {hindi_text}\nEnglish:"

    try:
        # Patched: Removed invalid timeout kwarg from text_generation
        response = client.text_generation(
            prompt,
            model="google/gemma-2b-it",
            max_new_tokens=50
        )
        return response.strip()
    except (TimeoutError, InferenceTimeoutError, HfHubHTTPError):
        raise LLMTimeoutError()
