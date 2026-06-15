import streamlit as st
from google import genai
from orchestration.error_handlers import LLMTimeoutError

def translate_hi_to_en(hindi_text: str) -> str:
    try:
        # Using the modern, official Google GenAI Client
        api_key = st.secrets.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=f"Translate the following Hindi agricultural query to English. Only provide the translation, no conversational text: {hindi_text}"
        )
        return response.text.strip()
    except Exception as e:
        raise Exception(f"Gemini API Error: {str(e)}")
