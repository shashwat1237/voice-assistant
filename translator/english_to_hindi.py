from huggingface_hub import InferenceClient
from huggingface_hub.errors import InferenceTimeoutError, HfHubHTTPError
import os
from orchestration.error_handlers import LLMTimeoutError

client = InferenceClient(token=os.environ.get("HF_TOKEN"), timeout=15.0)
PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "translator_prompt.txt")

def translate_en_to_hi(english_text: str) -> str:
    with open(PROMPT_PATH, "r") as f:
        system_prompt = f.read()

    prompt = f"{system_prompt}\n\nEnglish: {english_text}\nHindi:"

    try:
        # Patched: Removed invalid timeout kwarg. Token asymmetry (400) maintained.
        response = client.text_generation(
            prompt,
            model="google/gemma-2b-it",
            max_new_tokens=400
        )
        return response.strip()
    except (TimeoutError, InferenceTimeoutError, HfHubHTTPError):
        raise LLMTimeoutError()
