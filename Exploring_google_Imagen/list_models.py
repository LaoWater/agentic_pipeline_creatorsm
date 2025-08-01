import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the API key
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment variables")
    print("Please set your API key in your .env file or environment")
    exit(1)

genai.configure(api_key=api_key)

print("[*] Fetching list of available models...")

try:
    models = genai.list_models()

    print(f"Found {len(list(models))} models:")
    print("=" * 60)

    # Re-fetch since we consumed the generator above
    models = genai.list_models()

    for model in models:
        print(f"Model Name: {model.name}")
        print(f"  Display Name: {getattr(model, 'display_name', 'N/A')}")
        print(f"  Description: {getattr(model, 'description', 'N/A')}")
        print(f"  Supported Methods: {getattr(model, 'supported_generation_methods', 'N/A')}")
        print("-" * 40)

except Exception as e:
    print(f"Error fetching models: {e}")
    print("Make sure your API key is valid and you have internet connectivity")