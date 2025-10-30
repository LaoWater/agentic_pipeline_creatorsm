# image_control_service.py
from typing import Optional, Dict, Any
import requests
from io import BytesIO
from PIL import Image
import logging

from api_models import ImageControl, PlatformImageControl, ImageControlLevel1

logger = logging.getLogger(__name__)


class EffectiveImageControl:
    """Represents the effective image control settings for a specific platform."""
    
    def __init__(
        self,
        enabled: bool,
        style: str,
        guidance: str,
        caption: str,
        ratio: str,
        starting_image_url: Optional[str] = None,
        starting_image_path: Optional[str] = None
    ):
        self.enabled = enabled
        self.style = style
        self.guidance = guidance
        self.caption = caption
        self.ratio = ratio
        self.starting_image_url = starting_image_url
        self.starting_image_path = starting_image_path


class ImageControlProcessor:
    """Handles image control hierarchy and processing."""
    
    @staticmethod
    def resolve_effective_image_control(
        image_control: ImageControl,
        platform: str
    ) -> EffectiveImageControl:
        """
        Resolves the effective image control for a platform.
        Level 2 (platform-specific) always overrides Level 1 (global) when both exist.
        
        Args:
            image_control: The image control configuration
            platform: Platform name (facebook, instagram, linkedin, twitter)
            
        Returns:
            EffectiveImageControl: The resolved settings for the platform
        """
        level_1 = image_control.level_1
        level_2_controls = image_control.level_2
        
        # Get platform-specific control if it exists
        platform_control = None
        if level_2_controls:
            platform_control = getattr(level_2_controls, platform, None)
        
        # If platform-specific control exists and is enabled, use it
        if platform_control and platform_control.enabled:
            logger.info(f"Using Level 2 (platform-specific) image control for {platform}")
            return EffectiveImageControl(
                enabled=platform_control.enabled,
                style=platform_control.style,
                guidance=platform_control.guidance,
                caption=platform_control.caption,
                ratio=platform_control.ratio,
                starting_image_url=platform_control.starting_image_url
            )
        
        # Otherwise, use Level 1 (global) control
        logger.info(f"Using Level 1 (global) image control for {platform}")
        return EffectiveImageControl(
            enabled=level_1.enabled,
            style=level_1.style,
            guidance=level_1.guidance,
            caption=level_1.caption,
            ratio=level_1.ratio,
            starting_image_url=level_1.starting_image_url
        )
    
    @staticmethod
    async def download_starting_image(
        image_url: str,
        output_directory: str,
        filename_base: str
    ) -> Optional[str]:
        """
        Downloads a starting image from URL and saves it locally.
        
        Args:
            image_url: URL of the image to download
            output_directory: Directory to save the image
            filename_base: Base filename for the saved image
            
        Returns:
            str: Path to the downloaded image, or None if download failed
        """
        try:
            logger.info(f"Downloading starting image from: {image_url}")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Open image with PIL to validate and potentially convert format
            image = Image.open(BytesIO(response.content))
            
            # Save as PNG for consistency
            output_path = f"{output_directory}/{filename_base}_starting_image.png"
            image.save(output_path, "PNG")
            
            logger.info(f"Starting image downloaded and saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to download starting image from {image_url}: {e}")
            return None
    
    @staticmethod
    def sanitize_prompt_for_image_generation(prompt: str) -> str:
        """
        Sanitizes a prompt to prevent the image model from rendering instruction-like
        text or quoted phrases as overlays on the image.

        This function removes problematic patterns that models might misinterpret as
        text to be rendered on the image, such as:
        - Quoted phrases like 'for good' or "save the planet"
        - Instructional phrases like "emphasizing the X aspect"
        - Meta-commentary about what to convey

        Args:
            prompt: The raw prompt to sanitize

        Returns:
            str: Sanitized prompt with problematic patterns removed or transformed
        """
        import re

        # Remove common problematic instruction patterns
        problematic_patterns = [
            # Remove "emphasizing the 'X' aspect" or "highlighting the 'X' theme"
            r"emphasizing the ['\"][^'\"]+['\"] (aspect|theme|concept|message)",
            r"highlighting the ['\"][^'\"]+['\"] (aspect|theme|concept|message)",
            r"conveying (the message of |a sense of )?['\"][^'\"]+['\"]",
            r"with (the|a) ['\"][^'\"]+['\"] (feel|vibe|message|theme)",

            # Remove standalone quoted phrases that might be misinterpreted as text overlays
            # But be careful not to remove quotes that are part of legitimate descriptions
            r"emphasizing ['\"][^'\"]+['\"]",
            r"the ['\"][^'\"]+['\"] (aspect|element|feeling|theme|concept)",
        ]

        sanitized = prompt
        for pattern in problematic_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)

        # Clean up any double spaces or punctuation issues created by removals
        sanitized = re.sub(r'\s+', ' ', sanitized)  # Multiple spaces to single space
        sanitized = re.sub(r'\s*\.\s*\.', '.', sanitized)  # Double periods
        sanitized = re.sub(r',\s*,', ',', sanitized)  # Double commas
        sanitized = sanitized.strip()

        if sanitized != prompt:
            logger.info(f"Sanitized prompt - removed instruction-like text that could be misinterpreted")
            logger.debug(f"Original: {prompt[:100]}...")
            logger.debug(f"Sanitized: {sanitized[:100]}...")

        return sanitized

    @staticmethod
    def enhance_prompt_with_image_controls(
        base_prompt: str,
        effective_control: EffectiveImageControl,
        company_colors: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Enhances the base image generation prompt with image control settings.

        Args:
            base_prompt: The original prompt from the LLM
            effective_control: The resolved image control settings
            company_colors: Dictionary with primary_color_1 and primary_color_2

        Returns:
            str: Enhanced prompt with image control instructions
        """
        if not effective_control.enabled:
            # Still sanitize even if controls are not enabled
            return ImageControlProcessor.sanitize_prompt_for_image_generation(base_prompt)

        # First, sanitize the base prompt to remove problematic instruction-like text
        sanitized_prompt = ImageControlProcessor.sanitize_prompt_for_image_generation(base_prompt)

        enhanced_prompt = sanitized_prompt

        # CRITICAL: Add explicit instruction to NOT include text/captions/labels in the image
        enhanced_prompt += ". IMPORTANT: Create a clean visual design without any text, captions, labels, or written words in the image"

        # Add style guidance
        if effective_control.style:
            enhanced_prompt += f". Visual style: {effective_control.style}"

        # Add specific guidance
        if effective_control.guidance:
            enhanced_prompt += f". Creative direction: {effective_control.guidance}"

        # REMOVED: Caption is NOT added to prompt - captions should be post-generation metadata
        # Captions being in the prompt cause the AI to render them as text on the image

        # Add company colors in natural language (NOT hex codes or technical specifications)
        if company_colors:
            primary = company_colors.get('primary_color_1', '')
            secondary = company_colors.get('primary_color_2', '')

            # Convert color codes to natural language if they appear to be hex codes
            color_desc = self._describe_brand_colors(primary, secondary)
            if color_desc:
                enhanced_prompt += f". {color_desc}"

        # Add aspect ratio as a composition guide (not as text to render)
        # Valid ratios: "1:1", "3:4", "4:3", "9:16", "16:9" (Google Imagen supported values only)
        if effective_control.ratio:
            enhanced_prompt += f". Compose for {effective_control.ratio} aspect ratio"

        logger.info(f"Enhanced prompt: {enhanced_prompt[:200]}...")
        return enhanced_prompt

    @staticmethod
    def _describe_brand_colors(primary: str, secondary: str) -> str:
        """
        Converts color specifications into natural language descriptions for the AI.
        This prevents hex codes from being rendered as text on images.

        Args:
            primary: Primary color (hex code or color name)
            secondary: Secondary color (hex code or color name)

        Returns:
            str: Natural language color palette description
        """
        if not primary and not secondary:
            return ""

        # If colors are provided, describe them in a way that guides the AI's palette
        # without technical specifications that might be rendered as text
        color_parts = []

        if primary:
            # Check if it's a hex code
            if primary.startswith('#'):
                color_parts.append(f"incorporate the brand's primary color tone")
            else:
                color_parts.append(f"feature {primary} tones")

        if secondary:
            if secondary.startswith('#'):
                color_parts.append(f"with complementary accent colors")
            else:
                color_parts.append(f"complemented by {secondary}")

        return f"Color palette: {' '.join(color_parts)}"
    
    @staticmethod
    def get_image_generation_config(
        effective_control: EffectiveImageControl
    ) -> Dict[str, Any]:
        """
        Generates configuration parameters for image generation based on effective control.
        
        Args:
            effective_control: The resolved image control settings
            
        Returns:
            Dict: Configuration parameters for the image generation service
        """
        config = {}
        
        # Map ratio string to the value expected by the Google Imagen API
        # The API expects the ratio as a string, e.g., "1:1", "16:9"
        if effective_control.ratio:
            config["aspect_ratio"] = effective_control.ratio
        
        # Add starting image if available
        if effective_control.starting_image_path:
            config["starting_image_path"] = effective_control.starting_image_path
        
        return config