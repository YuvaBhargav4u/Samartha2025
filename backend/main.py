import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
import librosa
import torch
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
from transformers import pipeline

# --- Configuration ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

WHISPER_MODEL_NAME = "openai/whisper-base"
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'

# --- Model Loading (Done once at startup) ---
print("Loading models...")

whisper_pipeline = pipeline(
    "automatic-speech-recognition",
    model=WHISPER_MODEL_NAME,
    device_map="auto"
)

yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
yamnet_class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
yamnet_class_names = pd.read_csv(yamnet_class_map_path)['display_name'].tolist()

# --- MODIFIED LINE ---
# Changed "gemini-pro" to "gemini-pro-vision" for broader availability
llm = genai.GenerativeModel('gemini-2.5-flash')
# --- END MODIFIED LINE ---

print("Models loaded successfully!")

app = FastAPI()

# --- Helper Functions ---
def get_whisper_transcription(waveform, sample_rate):
    result = whisper_pipeline({"raw": waveform, "sampling_rate": sample_rate})
    return result["text"]

def get_yamnet_events(waveform, sample_rate):
    if sample_rate != 16000:
        waveform = librosa.resample(y=waveform, orig_sr=sample_rate, target_sr=16000)
    scores, _, _ = yamnet_model(waveform)
    scores = scores.numpy()
    top_scores_indices = np.mean(scores, axis=0).argsort()[-5:][::-1]
    events = [yamnet_class_names[i] for i in top_scores_indices if scores.mean(axis=0)[i] > 0.1]
    filtered_events = [e for e in events if e not in ["Speech", "Inside, small room", "Silence"]]
    return filtered_events

# --- API Endpoint ---
@app.post("/process-audio/")
async def process_audio(question: str = Form(...), audio_file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_audio_file:
            content = await audio_file.read()
            temp_audio_file.write(content)
            temp_audio_path = temp_audio_file.name

        waveform, sr = librosa.load(temp_audio_path, sr=16000)

        whisper_text = get_whisper_transcription(waveform, sr)
        yamnet_events = get_yamnet_events(waveform, sr)

        context = f"""
        ### Audio Analysis Report
        **Audio Transcription (from Whisper):**
        "{whisper_text if whisper_text else 'No speech detected.'}"

        **Detected General Sounds (from YAMNet):**
        - {', '.join(yamnet_events) if yamnet_events else 'None detected.'}
        """

        prompt = f"""
        You are an intelligent audio analysis assistant. Based *only* on the provided "Audio Analysis Report", answer the user's question.
        Do not make up information not present in the report.
        {context}
        ---
        **User's Question:** {question}
        **Answer:**
        """

        response = llm.generate_content(prompt)

        return {"answer": response.text}

    finally:
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)