# visual_model.py

import os
from pathlib import Path
from io import BytesIO
import logging
from typing import Optional, Dict, Any


# --- Google Imagen Imports ---
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
from google.genai.types import GenerateImagesConfig
from PIL import Image
from dotenv import dotenv_values

# Import centralized model config
from config import IMAGE_GENERATION_MODEL

# --- Setup Logging ---
logger = logging.getLogger(__name__)


# --- Google Imagen Configuration ---
_google_genai_client = None
_gemini_api_key_loaded = False


def _load_gemini_api_key() -> str:
    """Loads GEMINI_API_KEY from .env or environment variables."""
    print("[INFO] Loading GEMINI_API_KEY")

    # Load .env into a local config dict (does not modify os.environ)
    config = dotenv_values(".env")

    # 1. Try .env
    api_key = config.get("GEMINI_API_KEY")
    if api_key:
        print("[INFO] Using GEMINI_API_KEY from .env")
        return api_key

    # 2. Try environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        print("[INFO] Using GEMINI_API_KEY from environment")
        return api_key

    # 3. Raise error if not found
    raise ValueError("GEMINI_API_KEY is not set in .env or environment variables.")



def get_google_genai_client():
    """Initializes and returns the Google GenAI client."""
    global _google_genai_client
    api_key = _load_gemini_api_key()

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables or .env file. "
            "Please set it to use Google Imagen."
        )
    if _google_genai_client is None:
        try:
            print("Init Google GenAi")
            # Explicitly pass the API key to ensure it uses the one from .env
            _google_genai_client = genai.Client(api_key=api_key)
            logger.info("Google GenAI Client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI Client: {e}")
            raise
    return _google_genai_client


def generate_and_save_image_google(
        prompt: str,
        output_directory: str,
        filename_base: str,
        file_extension: str = "png",
        model_name: str = IMAGE_GENERATION_MODEL,  # Centralized model config from config.py
        image_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generates an image using a specified Google Imagen model and saves it.
    This function is synchronous and should be called via asyncio.to_thread() for async contexts.

    Args:
        prompt: Text prompt for image generation.
        output_directory: Directory where the generated asset should be saved.
        filename_base: Base filename (without extension) for the generated asset.
        file_extension: File extension for the generated image (e.g., "png", "jpg").
        model_name: The specific Google Imagen model to use.
        image_config: Optional dictionary that specifies how the generated asset should be

    Returns:
        The file path where the image was saved.

    Raises:
        ValueError: If GEMINI_API_KEY is not set.
        Exception: If image generation or saving fails.
    """
    try:
        client = get_google_genai_client()  # Ensures client is initialized and API key is present
    except ValueError as ve:  # Catch API key error specifically
        logger.error(f"Configuration error for Google Imagen: {ve}")
        raise  # Re-raise to be handled by the caller

    logger.info(
        f"🎨 [Google Imagen] Requesting image generation with model '{model_name}' for prompt: '{prompt[:100]}...'")

    try:
        # Prepare image generation config
        gen_config_args = {"number_of_images": 1}
        if image_config and "aspect_ratio" in image_config:
            aspect_ratio = image_config["aspect_ratio"]
            # Validate aspect_ratio against allowed values if necessary
            # For example: if aspect_ratio not in ["square", "portrait", "landscape"]: ...
            gen_config_args["aspect_ratio"] = aspect_ratio
            logger.info(f"🖼️ [Google Imagen] Using aspect ratio: {aspect_ratio}")

        response = client.models.generate_images(
            model=model_name,
            prompt=prompt,
            config=types.GenerateImagesConfig(**gen_config_args)
        )

        if not response.generated_images:
            raise Exception("Google Imagen API did not return any images.")

        # Get the first generated image's data
        generated_image_data = response.generated_images[0]
        if not generated_image_data.image or not generated_image_data.image.image_bytes:
            raise Exception("Google Imagen API returned an image object without image_bytes.")

        image_bytes = generated_image_data.image.image_bytes
        pil_image = Image.open(BytesIO(image_bytes))

        # Create output directory if it doesn't exist
        Path(output_directory).mkdir(parents=True, exist_ok=True)

        # Construct the full file path
        # Ensure file_extension does not start with a dot and is lowercase
        clean_file_extension = file_extension.lstrip('.').lower()
        if not clean_file_extension:  # Default to png if empty
            clean_file_extension = "png"

        file_path = Path(output_directory) / f"{filename_base}.{clean_file_extension}"

        # Determine PIL format string (e.g., 'PNG', 'JPEG')
        pil_format = clean_file_extension.upper()
        if pil_format == 'JPG':
            pil_format = 'JPEG'
        # Add more format mappings if needed for other extensions like WEBP

        pil_image.save(file_path, format=pil_format)
        logger.info(f"🖼️ [Google Imagen] Image saved successfully to: {file_path}")
        logger.info(f"🖼️ [Google Imagen] Image size: {len(image_bytes)} bytes")
        return str(file_path)

    except google_exceptions.GoogleAPICallError as e:  # Catch Google specific API errors
        error_msg = f"Google Imagen API error for '{filename_base}' (model '{model_name}'): {e}"
        logger.error(f"🚨 [Google Imagen] {error_msg}")
        raise Exception(error_msg) from e  # Preserve original exception context
    except Exception as e:
        error_msg = f"Google Imagen generation or saving failed for '{filename_base}' (model '{model_name}'): {e}"
        logger.error(f"🚨 [Google Imagen] {error_msg}")
        raise Exception(error_msg) from e


# Test function for standalone usage of this module (optional)
if __name__ == "__main__":

    print("Hello from Main")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("🚀 Testing Google Imagen Visual Model Integration...")

    # IMPORTANT: Ensure your .env file with GEMINI_API_KEY is in the correct location
    # or GEMINI_API_KEY is set as an environment variable.
    # Example .env content:
    # GEMINI_API_KEY="YOUR_API_KEY_HERE"

    test_prompt = "A charming robot tending a small, vibrant flower garden on a sunny day, digital art style."
    test_output_dir = "test_generated_images_google"
    test_filename = "test_google_image"

    try:
        print("Init Generate Image")
        result_path = generate_and_save_image_google(
            prompt=test_prompt,
            output_directory=test_output_dir,
            filename_base=test_filename,
            file_extension="png",
            model_name=IMAGE_GENERATION_MODEL  # Using centralized model config
        )
        logger.info(f"✅ Google Imagen Test successful! Image saved to: {result_path}")
    except ValueError as ve:
        logger.error(f"❌ Configuration Error: {ve}. Please ensure GEMINI_API_KEY is set correctly.")
    except Exception as e:
        logger.error(f"❌ Google Imagen Test failed: {e}")

    logger.info("\n🏁 Google Imagen Visual Model Test Finished.")