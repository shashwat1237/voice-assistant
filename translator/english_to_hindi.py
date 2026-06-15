import streamlit as st
import google.generativeai as genai
from orchestration.error_handlers import LLMTimeoutError

def translate_en_to_hi(english_text: str) -> str:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Translate the following English agricultural text to Hindi. Only provide the translation, no conversational text: {english_text}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
         raise LLMTimeoutError()
