from gtts import gTTS

def text_to_speech(text: str, output_path: str) -> str:
    if not text:
        return ""
    try:
        tts = gTTS(text=text, lang='hi', slow=False)
        tts.save(output_path)
        return output_path
    except Exception:
        return ""
