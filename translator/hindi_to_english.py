from deep_translator import GoogleTranslator
from orchestration.error_handlers import LLMTimeoutError

def translate_hi_to_en(hindi_text: str) -> str:
    try:
        # Instant translation via Google engine
        translated = GoogleTranslator(source='hi', target='en').translate(hindi_text)
        return translated.strip()
    except Exception:
        raise LLMTimeoutError()
