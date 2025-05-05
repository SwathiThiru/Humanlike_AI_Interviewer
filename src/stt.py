import whisper

# load once on startup
model = whisper.load_model("base")

def transcribe(audio_path: str) -> str:
    res = model.transcribe(audio_path)
    return res["text"].strip()