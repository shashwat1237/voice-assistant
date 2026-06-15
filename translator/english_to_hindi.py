import google.generativeai as genai
import os
from orchestration.error_handlers import LLMTimeoutError

def translate_en_to_hi(english_text: str) -> str:
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Translate the following English agricultural text to Hindi. Only provide the translation, no conversational text: {english_text}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
         raise LLMTimeoutError()
