import streamlit as st
import os
import uuid
import requests
from audio_recorder_streamlit import audio_recorder
from voice.whisper import speech_to_text
from voice.tts import text_to_speech
from translator.hindi_to_english import translate_hi_to_en
from translator.english_to_hindi import translate_en_to_hi
from orchestration.router import route_query
from orchestration.state_manager import init_session_state, get_history_string, update_history, clear_history
from orchestration.error_handlers import ERROR_MESSAGES, EmptyAudioError, NoisyAudioError, PartialRecordingError, LLMTimeoutError, VectorDBError, OutOfDomainError

st.set_page_config(page_title="किसान सहायक (Farmer Assistant)", layout="centered")

init_session_state()

st.title("🌾 किसान सहायक (Natural Farming Assistant)")

# The Third-Party Live Mic Bypass
st.write("🎤 अपना प्रश्न रिकॉर्ड करें (Tap mic to record)")
audio_bytes = audio_recorder(text="", icon_size="2x")

text_input = st.text_input("या अपना प्रश्न यहाँ लिखें (Or type here):")

col1, col2 = st.columns([1, 5])
with col1:
    ask_button = st.button("पूछें (Ask)")
with col2:
    if st.button("संदर्भ रीसेट करें (Clear Memory)"):
        clear_history()
        st.success("मेमोरी साफ़ हो गई। (Memory cleared.)")

# Auto-trigger if audio is recorded, OR if text is typed and Ask is clicked
if audio_bytes or (ask_button and text_input):
    hindi_query = ""
    temp_audio_path = None
    out_audio_path = None
    st.markdown("---")
    
    try:
        # File teardown and bytes writing
        if audio_bytes:
            temp_audio_path = f"temp_{uuid.uuid4().hex}.wav"
            with open(temp_audio_path, "wb") as f:
                f.write(audio_bytes)
            with st.spinner("आवाज़ को टेक्स्ट में बदला जा रहा है (Processing Audio)..."):
                hindi_query = speech_to_text(temp_audio_path)
                st.info(f"📝 आपकी आवाज़ (You said): {hindi_query}")
        elif text_input:
            hindi_query = text_input
            
        if not hindi_query:
            raise EmptyAudioError()

        with st.spinner("सिस्टम सोच रहा है (Thinking)..."):
            english_query = translate_hi_to_en(hindi_query)
            current_history = get_history_string()
            
            rag_response = route_query(english_query, current_history)
            
            if rag_response["answer"] == "Please specify your question.":
                 st.warning(ERROR_MESSAGES["missing_context"])
                 st.stop()
                 
            hindi_answer = translate_en_to_hi(rag_response["answer"])
            update_history(english_query, rag_response["answer"])
            
            out_audio_path = f"out_{uuid.uuid4().hex}.mp3"
            audio_out = text_to_speech(hindi_answer, out_audio_path)

        st.markdown("### 🌾 उत्तर (Answer):")
        st.write(hindi_answer)
        
        st.markdown("---")
        
        if audio_out and os.path.exists(audio_out):
            with open(audio_out, "rb") as audio_file_reader:
                out_bytes = audio_file_reader.read()
            st.audio(out_bytes, format="audio/mp3")
        
        st.markdown("### 🔍 उपयोग किए गए स्रोत (Sources Used):")
        for source in rag_response["sources"]:
            st.write(f"- {source}")
            
        if rag_response["context_used"]:
            with st.expander("विस्तृत संदर्भ देखें (View Context)"):
                for chunk in rag_response["context_used"]:
                    st.write(chunk)
                    st.markdown("---")

    except EmptyAudioError:
        st.error(ERROR_MESSAGES["empty_recording"])
    except NoisyAudioError:
        st.error(ERROR_MESSAGES["audio_unclear"])
    except PartialRecordingError:
        st.error(ERROR_MESSAGES["partial_recording"])
    except LLMTimeoutError:
        st.error(ERROR_MESSAGES["llm_timeout"])
    except VectorDBError:
        st.error(ERROR_MESSAGES["db_failure"])
    except OutOfDomainError:
        st.error(ERROR_MESSAGES["info_unavailable"])
    except requests.exceptions.ConnectionError:
        st.error(ERROR_MESSAGES["no_internet"])
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if out_audio_path and os.path.exists(out_audio_path):
            os.remove(out_audio_path)
