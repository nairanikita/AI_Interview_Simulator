import os
import tempfile
from io import BytesIO

import whisper
from gtts import gTTS

# loaded once when the module is first imported so every request reuses it
# downloading "base" model is ~140MB and happens only on first run
_model = whisper.load_model("base")


def transcribe_audio(audio_bytes: bytes) -> str:
    """Convert raw audio bytes to a transcript string using Whisper.

    Returns an empty string if transcription fails for any reason,
    so the caller never has to handle exceptions from this function.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        result = _model.transcribe(tmp_path)
        return result["text"].strip()

    except Exception:
        return ""

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def text_to_speech(text: str) -> bytes:
    """Convert text to MP3 bytes using gTTS, ready for st.audio()."""
    tts = gTTS(text=text, lang="en", slow=False)
    buffer = BytesIO()
    tts.write_to_fp(buffer)
    return buffer.getvalue()
