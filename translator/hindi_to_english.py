from deep_translator import MyMemoryTranslator
from orchestration.error_handlers import LLMTimeoutError

def translate_hi_to_en(hindi_text: str) -> str:
    try:
        translated = MyMemoryTranslator(source='hi', target='en').translate(hindi_text)
        return translated.strip()
    except Exception:
        raise LLMTimeoutError()
