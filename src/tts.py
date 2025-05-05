from gtts import gTTS
import tempfile

def text_to_speech(text: str) -> str:
    """Saves TTS mp3 to a temp file and returns its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    gTTS(text).save(tmp.name)
    return tmp.name