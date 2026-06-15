class VoiceAssistantError(Exception):
    pass
class EmptyAudioError(VoiceAssistantError):
    pass
class NoisyAudioError(VoiceAssistantError):
    pass
class PartialRecordingError(VoiceAssistantError):
    pass
class LLMTimeoutError(VoiceAssistantError):
    pass
class VectorDBError(VoiceAssistantError):
    pass
class OutOfDomainError(VoiceAssistantError):
    pass

ERROR_MESSAGES = {
    "empty_recording": " कृपया फिर से रिकॉर्ड करें। (Please record again.)",
    "audio_unclear": " आवाज़ स्पष्ट नहीं है। कृपया फिर से बोलें। (Audio unclear. Please speak again.)",
    "partial_recording": " पूरा प्रश्न समझ नहीं सका। (Could not understand complete question.)",
    "info_unavailable": " जानकारी उपलब्ध नहीं है। (Information unavailable.)",
    "missing_context": " कृपया अपना प्रश्न स्पष्ट करें। (Please specify your question.)",
    "db_failure": " डेटाबेस से संपर्क टूट गया है। (Database connection lost.)",
    "llm_timeout": " अनुरोध में अपेक्षा से अधिक समय लग रहा है। कृपया पुनः प्रयास करें। (Request taking longer than expected. Please try again.)",
    "no_internet": " सर्वर से कनेक्ट करने में असमर्थ। (Unable to connect to server.)"
}
