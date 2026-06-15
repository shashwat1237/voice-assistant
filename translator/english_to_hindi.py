import streamlit as st
from google import genai
from orchestration.error_handlers import LLMTimeoutError

def translate_en_to_hi(english_text: str) -> str:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=f"Translate the following English agricultural text to Hindi. Only provide the translation, no conversational text: {english_text}"
        )
        return response.text.strip()
    except Exception as e:
         raise Exception(f"Gemini API Error: {str(e)}")
