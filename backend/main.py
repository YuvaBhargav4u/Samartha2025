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
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

# --- Configuration ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

BASE_WHISPER_MODEL = "openai/whisper-base"
FINETUNED_ADAPTER_PATH = "./whisper_finetuned_speech"
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'

# --- Model Loading ---
print("Loading models...")

base_model = WhisperForConditionalGeneration.from_pretrained(
    BASE_WHISPER_MODEL, load_in_8bit=True, device_map="auto"
)
whisper_model = PeftModel.from_pretrained(base_model, FINETUNED_ADAPTER_PATH)
whisper_processor = WhisperProcessor.from_pretrained(BASE_WHISPER_MODEL, language="english", task="transcribe")

yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
yamnet_class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
yamnet_class_names = pd.read_csv(yamnet_class_map_path)['display_name'].tolist()

# Use a model confirmed to work with your key
llm = genai.GenerativeModel('gemini-2.5-flash') # Or 'gemini-pro'

print("All models loaded successfully!")

app = FastAPI()

# --- Helper Functions ---
def get_whisper_description(waveform, sample_rate):
    """Generates a description using the fine-tuned Whisper model."""
    inputs = whisper_processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(whisper_model.device, dtype=torch.float16)
    with torch.no_grad():
        generated_ids = whisper_model.generate(inputs=input_features, max_new_tokens=100)
    description = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return description

def get_yamnet_events(waveform, sample_rate):
    """Detects general audio events using YAMNet."""
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

        whisper_desc = get_whisper_description(waveform, sr)
        yamnet_events = get_yamnet_events(waveform, sr)

        context = f"""
        ### Audio Analysis Report
        **Speech Transcription:** "{whisper_desc if whisper_desc else 'No speech detected.'}"
        **Detected Background Sounds:** - {', '.join(yamnet_events) if yamnet_events else 'None detected.'}
        """

        # --- MODIFIED AND IMPROVED PROMPT ---
        prompt = f"""
        You are an intelligent audio analysis assistant. Your task is to analyze the provided report and answer the user's question, by keeping in mind that the audio may only contain languages from the following list: arabic,english,hebrew,hindi,indonesian,russian,telugu,thai.

        **Instructions:**
        1.  Analyze the "Speech Transcription". It may contain special tags like `[Language: ...]` and `[Speaker: ...]`.
        2.  If the language is NOT English, provide an English translation of the transcription.
        3.  Use all the information in the report to give a comprehensive answer to the user's question.

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