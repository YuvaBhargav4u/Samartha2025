#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# 1. Start the FastAPI backend server IN THE BACKGROUND.
# The '&' symbol is critical. It runs this command, then immediately
# moves to the next line without waiting for this one to finish.
# We host it on port 8000.
echo "Starting backend server on port 8000..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# 2. Start the Streamlit frontend app IN THE FOREGROUND.
# This script will "wait" on this line. Render's health check will
# ping the $PORT. When this app starts, the service is "live".
# The streamlit app will connect to http://127.0.0.1:8000 to talk to the backend.
echo "Starting frontend server on port $PORT..."
streamlit run ui/app.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false
