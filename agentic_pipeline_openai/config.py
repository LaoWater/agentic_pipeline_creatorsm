# config.py
from dotenv import dotenv_values
import os

# --- API Keys & Environment ---
config = dotenv_values(".env")


def get_openai_api_key(config: dict) -> str:
    # 1. Try config dict
    key = config.get("OPENAI_API_KEY_VALUE")
    if key:
        print("[INFO] Using OPENAI_API_KEY_VALUE from config.")
        return key

    # 2. Try environment variable
    key = os.environ.get("OPENAI_API_KEY_VALUE")
    if key:
        print("[INFO] Using OPENAI_API_KEY_VALUE from environment.")
        return key

    # 3. Fallback failed
    raise ValueError("Missing OPENAI_API_KEY_VALUE. Set it in config or as an environment variable.")



OPENAI_API_KEY_VALUE = get_openai_api_key(config)


# --- LLM Model Configuration ---
DECISION_LLM_MODEL = "gpt-4o"
PLATFORM_LLM_MODEL = "gpt-4.1-2025-04-14"

# --- Output Configuration ---
if os.environ.get("RUNNING_IN_DOCKER") == "true":
    BASE_OUTPUT_FOLDER = "/app/output_data"
    print(f"RUNNING IN DOKCER: True, BASE_OUTPUT_FOLDER: {BASE_OUTPUT_FOLDER}")
else:
    BASE_OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_data_local")
    print(f"RUNNING IN DOKCER: False, BASE_OUTPUT_FOLDER: {BASE_OUTPUT_FOLDER}")



# # --- Company & Request Configuration ---
# COMPANY_NAME = "Creators Multiverse"
# COMPANY_MISSION = "Empowering creators to build their digital presence with AI-powered tools that transform ideas into viral content across platforms"
# COMPANY_SENTIMENT = ("Inspirational & Empowering. Cosmic/Magical Theme yet not too much."
#                      "The brand positions itself as a creative partner that amplifies human creativity rather than replacing it.")
# # This will be passed to the main orchestrator, but a default can be here
# DEFAULT_POST_SUBJECT = "Hello World! Intro post about our company, starting out, vision, etc"
#
# LANGUAGE = "English"
#
# TONE = 'Casual/Friendly'
#
# PLATFORMS_POST_TYPES_MAP = [
#     {"linkedin": "Image"}, # "Text", "Image", or "Video"
#     # {"facebook": "Text"},
#     {"twitter": "Text"},
#     {"instagram": "Text"}
# ]