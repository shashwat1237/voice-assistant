import streamlit as st
from groq import Groq
from orchestration.error_handlers import LLMTimeoutError

def translate_en_to_hi(english_text: str) -> str:
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        client = Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a direct translator. Translate the following English agricultural text to Hindi. Only provide the translation in Devnagari script, no conversational text."},
                {"role": "user", "content": english_text}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
         raise Exception(f"Groq API Error: {str(e)}")
