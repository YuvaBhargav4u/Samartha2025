#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Start the FastAPI server in the background
uvicorn backend/main:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit app
# Note: Render's health check will ping the service at the port specified by the PORT env var.
# Streamlit must run on this port.
streamlit run ui/app.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false
