import streamlit as st
import google.generativeai as genai
from orchestration.error_handlers import LLMTimeoutError

def translate_hi_to_en(hindi_text: str) -> str:
    try:
        # Fetching directly from Streamlit's secrets manager
        api_key = st.secrets.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Translate the following Hindi agricultural query to English. Only provide the translation, no conversational text: {hindi_text}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        raise LLMTimeoutError()
