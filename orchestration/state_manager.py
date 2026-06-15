import streamlit as st

def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def get_history_string() -> str:
    if not st.session_state.chat_history:
        return "No prior context."
    formatted_history = ""
    for interaction in st.session_state.chat_history[-3:]:
        formatted_history += f"User: {interaction['user']}\nSystem: {interaction['system']}\n"
    return formatted_history

def update_history(user_query: str, system_response: str):
    st.session_state.chat_history.append({
        "user": user_query,
        "system": system_response
    })

def clear_history():
    st.session_state.chat_history = []
