import streamlit as st
from faster_whisper import WhisperModel
import os
from orchestration.error_handlers import EmptyAudioError, NoisyAudioError, PartialRecordingError

@st.cache_resource
def load_whisper_model():
    return WhisperModel("small", device="cpu", compute_type="int8")

def speech_to_text(audio_path: str) -> str:
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
        raise EmptyAudioError()

    model = load_whisper_model()

    # Patched: Explicit Voice Activity Detection (VAD) to prevent hallucinating noise
    segments, info = model.transcribe(
        audio_path,
        language="hi",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    transcript = " ".join([segment.text for segment in segments]).strip()

    if not transcript:
        raise EmptyAudioError()

    # Detect prematurely cut-off speech
    if len(transcript.split()) < 2:
        raise PartialRecordingError()

    return transcript
