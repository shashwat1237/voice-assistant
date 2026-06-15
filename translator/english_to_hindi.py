from deep_translator import MyMemoryTranslator
from orchestration.error_handlers import LLMTimeoutError

def translate_en_to_hi(english_text: str) -> str:
    try:
        translated = MyMemoryTranslator(source='en', target='hi').translate(english_text)
        return translated.strip()
    except Exception:
         raise LLMTimeoutError()
