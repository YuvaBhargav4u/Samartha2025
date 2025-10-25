import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load the API key from your .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("Available generative models:")
print("-" * 30)

# List all models and check if they support the 'generateContent' method
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(f"- {m.name}")

print("-" * 30)
print("\nRecommendation: Copy one of the model names from the list above and paste it into your backend/main.py file.")