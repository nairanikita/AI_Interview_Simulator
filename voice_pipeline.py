import os
import tempfile
from io import BytesIO

from gtts import gTTS

# Whisper is only loaded if transcribe_audio() is actually called.
# The main app uses browser-native STT so this stays None in normal use.
_whisper_model = None


def transcribe_audio(audio_bytes: bytes) -> str:
    """Fallback STT using local Whisper — only used outside the browser flow."""
    global _whisper_model
    tmp_path = None
    try:
        if _whisper_model is None:
            import whisper
            _whisper_model = whisper.load_model("base")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        result = _whisper_model.transcribe(tmp_path)
        return result["text"].strip()

    except Exception:
        return ""

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def text_to_speech(text: str) -> bytes:
    """Convert text to MP3 bytes using gTTS, ready for playback."""
    tts = gTTS(text=text, lang="en", slow=False)
    buffer = BytesIO()
    tts.write_to_fp(buffer)
    return buffer.getvalue()
