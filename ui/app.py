import streamlit as st
import requests

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000/process-audio/"

# --- UI Layout ---
st.set_page_config(layout="wide")
st.title("🔊 Audio Scene Q&A")
st.markdown("Upload an audio clip, ask a question about it, and get an AI-powered answer.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Your Audio")
    uploaded_file = st.file_uploader("Choose a WAV, MP3, or M4A file", type=["wav", "mp3", "m4a"])
    
    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')

with col2:
    st.subheader("2. Ask a Question")
    question = st.text_input("e.g., What did the person say before the bell rang?")
    
    submit_button = st.button("Analyze and Answer")

# --- Logic to call backend ---
if submit_button and uploaded_file is not None and question:
    with st.spinner('Analyzing audio... Please wait.'):
        try:
            # Prepare the data for the POST request
            files = {'audio_file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            payload = {'question': question}
            
            # Send the request to the FastAPI backend
            response = requests.post(BACKEND_URL, files=files, data=payload)
            
            if response.status_code == 200:
                answer = response.json().get("answer")
                st.subheader("💡 Answer")
                st.success(answer)
            else:
                st.error(f"Error from server: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the backend. Please ensure it's running. Error: {e}")

elif submit_button:
    st.warning("Please upload an audio file and enter a question.")