# media_generation.py
import asyncio
import logging
from typing import Literal, Optional, Dict, Any

# Import the Google Imagen specific function from visual_model.py
from visual_model import generate_and_save_image_google

from config import IMAGE_FILE_EXTENSION, IMAGE_GENERATION_MODEL

# Default configuration - adjust as needed

# --- Setup Logging ---
# Using the root logger. Configure it in your main application entry point.
# For example, in main(): logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


async def generate_visual_asset_for_platform(
        image_prompt: str,
        output_directory: str,  # e.g., "generated_posts/facebook"
        filename_base: str,  # e.g., "hello_world_intro_post"
        media_type: str,  # Extend for "video" later
        image_config: Optional[Dict[str, Any]] = None,  # NEW: For aspect ratio, etc.
        model: str = IMAGE_GENERATION_MODEL,  # Centralized model config from config.py
        file_extension: str = IMAGE_FILE_EXTENSION
) -> str:
    """
    Orchestrates the generation and saving of a visual asset using Google Imagen via visual_model.

    This function runs the synchronous generate_and_save_image_google in a separate thread
    to avoid blocking the asyncio event loop.

    Args:
        image_prompt: Text prompt for image generation.
        output_directory: Directory where the generated asset should be saved.
        filename_base: Base filename (without extension) for the generated asset.
        media_type: Type of media to generate (currently only "image" supported).
        model: Google Imagen model to use for generation.
        file_extension: File extension for the generated image.
        image_config: Optional dict with configuration settings for image generation.

    Returns:
        The file path where the media was saved.

    Raises:
        NotImplementedError: If media_type is not "image".
        Exception: If image generation fails (e.g., API key issue, API error).
    """
    logger.info(f"\n🖼️ Media Generation Task: Starting for '{filename_base}' in '{output_directory}' using Google Imagen.")
    logger.info(f"🖼️ Media Generation Task: Using prompt: '{image_prompt}...' with model '{model}'")

    if media_type == "image":
        try:
            # Run the synchronous visual_model.generate_and_save_image_google in a thread
            file_path = await asyncio.to_thread(
                generate_and_save_image_google,  # Call the Google Imagen function
                prompt=image_prompt,
                output_directory=output_directory,
                filename_base=filename_base,
                file_extension=file_extension,
                model_name=model,  # Pass the model to the 'model_name' parameter
                image_config=image_config  # NEW: Pass the image config
            )

            logger.info(f"✅ Media Generation Task: Asset ready at {file_path}")
            return file_path

        except Exception as e:
            # Error details should be logged by generate_and_save_image_google
            error_msg = f"Error during visual asset generation for '{filename_base}': {e}"
            logger.error(f"🚨 Media Generation Task: {error_msg}")
            # Re-raise the exception to make errors visible to the calling code
            raise Exception(error_msg) from e # Preserve original exception for traceback

    else:
        # Placeholder for video or other media types
        error_msg = f"Media type '{media_type}' not yet supported for generation."
        logger.warning(f"⚠️ Media Generation Task: {error_msg}")
        raise NotImplementedError(error_msg)


async def generate_multiple_visual_assets(
        prompts_and_configs: list[dict],
        base_output_directory: str = "generated_posts_google" # Default changed for clarity
) -> list[str]:
    """
    Generate multiple visual assets concurrently using Google Imagen.

    Args:
        prompts_and_configs: List of dictionaries containing:
            - prompt: Image generation prompt
            - platform: Platform name (e.g., "facebook", "instagram")
            - filename_base: Base filename
            - media_type: Type of media (default: "image")
            - model: (Optional) Specific Google Imagen model for this task
                       (defaults to "imagen-3.0-generate-002" if not provided)
            - file_extension: (Optional) Specific file extension for this task
        base_output_directory: Base directory for all generated assets

    Returns:
        List of file paths where assets were saved.
    """
    logger.info(f"\n🎬 Batch Media Generation: Starting {len(prompts_and_configs)} tasks using Google Imagen...")

    tasks = []
    for config in prompts_and_configs:
        platform_name = config.get('platform', 'default_platform')
        platform_dir = f"{base_output_directory}/{platform_name}"

        task = generate_visual_asset_for_platform(
            image_prompt=config['prompt'],
            output_directory=platform_dir,
            filename_base=config['filename_base'],
            media_type=config.get('media_type', 'image'),
            model=config.get('model', IMAGE_GENERATION_MODEL),  # Allow per-task model override, default to centralized config
            file_extension=config.get('file_extension', IMAGE_FILE_EXTENSION)
        )
        tasks.append(task)

    try:
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_paths = []
        for i, result in enumerate(results):
            config_info = prompts_and_configs[i] # For better error reporting
            filename_info = config_info.get('filename_base', f"task_{i+1}")
            platform_info = config_info.get('platform', 'default')

            if isinstance(result, Exception):
                logger.error(f"❌ Batch Task for '{filename_info}' (Platform: {platform_info}) failed: {result}")
            elif result is None: # Should ideally not happen if exceptions are raised properly
                 logger.error(f"❌ Batch Task for '{filename_info}' (Platform: {platform_info}) returned None unexpectedly.")
            else:
                successful_paths.append(result)
                logger.info(f"✅ Batch Task for '{filename_info}' (Platform: {platform_info}) completed: {result}")

        logger.info(
            f"\n🎬 Batch Media Generation: Completed {len(successful_paths)}/{len(prompts_and_configs)} tasks successfully")
        return successful_paths

    except Exception as e:
        logger.critical(f"🚨 Batch Media Generation: Unexpected critical error during task gathering: {e}")
        raise


# --- Demo/Test Functions ---
async def demo_single_generation():
    """Demo function for testing single image generation with Google Imagen."""
    logger.info("🧪 Demo: Single Image Generation (Google Imagen)")

    try:
        file_path = await generate_visual_asset_for_platform(
            image_prompt="A professional social media post background with modern gradient colors, designed using Google Imagen.",
            output_directory="demo_output_google",
            filename_base="demo_post_bg_google",
            model=IMAGE_GENERATION_MODEL  # Using centralized model config
        )
        logger.info(f"Single demo (Google Imagen) completed successfully: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Single demo (Google Imagen) failed: {e}")
        return None


async def demo_batch_generation():
    """Demo function for testing batch image generation with Google Imagen."""
    logger.info("🧪 Demo: Batch Image Generation (Google Imagen)")

    configs = [
        {
            'prompt': 'A vibrant Instagram-style food photo with natural lighting, using Imagen 3',
            'platform': 'instagram',
            'filename_base': 'food_post_1_google'
        },
        {
            'prompt': 'A professional LinkedIn banner with a subtle business theme and abstract geometric patterns, Google Imagen',
            'platform': 'linkedin',
            'filename_base': 'business_banner_google',
            'model': IMAGE_GENERATION_MODEL  # Example of per-task model, can be different or use default
        },
        {
            'prompt': 'A fun Facebook cover photo with diverse community members interacting joyfully, illustration style, Imagen',
            'platform': 'facebook',
            'filename_base': 'community_cover_google',
            'file_extension': 'jpg' # Example of per-task file extension
        }
    ]

    try:
        results = await generate_multiple_visual_assets(configs, "demo_batch_output_google")
        logger.info(f"Batch demo (Google Imagen) completed: {len(results)} assets generated.")
        return results
    except Exception as e:
        logger.error(f"Batch demo (Google Imagen) failed: {e}")
        return []


# --- Main execution for testing ---
async def main():
    """Main function for testing the Google Imagen media generation pipeline."""
    # Setup basic logging for the demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)-8s - %(name)-15s - %(message)s'
    )
    logger.info("🚀 Google Imagen Media Generation Pipeline - Test Mode")

    # IMPORTANT: Ensure your .env file with GEMINI_API_KEY is in the correct location
    # (usually project root or same directory as visual_model.py)
    # or GEMINI_API_KEY is set as an environment variable.
    # Example .env content:
    # GEMINI_API_KEY="YOUR_ACTUAL_API_KEY_HERE"

    # Test single generation
    await demo_single_generation()

    logger.info("\n" + "=" * 60 + "\n")

    # Test batch generation
    await demo_batch_generation()

    logger.info("\n🏁 All Google Imagen demos completed!")


if __name__ == "__main__":
    asyncio.run(main())