# api_models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union


# Re-use from your existing data_models if they fit, or define specific for API
class RequirementItem(BaseModel):  # Example, adapt from your Requirements TypedDict
    type: str
    detail: str


class PostHistoryItem(BaseModel):  # Example, adapt from your PostHistoryEntry
    platform: str
    text: str
    # ... other fields


# NEW: Enhanced models for ContentGeneratorData structure
class Company(BaseModel):
    id: str = Field(..., description="Unique identifier for the company")
    name: str = Field(..., description="Company name")
    mission: str = Field(..., description="Company mission statement")
    tone_of_voice: str = Field(..., description="Company's tone of voice")
    primary_color_1: str = Field(..., description="Primary brand color (hex format)")
    primary_color_2: str = Field(..., description="Secondary brand color (hex format)")
    logo_path: Optional[str] = Field(None, description="Path to company logo file")


class Content(BaseModel):
    topic: str = Field(..., description="Main topic for the content")
    description: str = Field(..., description="Detailed description of the content")
    hashtags: List[str] = Field(..., description="List of hashtags to include")
    call_to_action: str = Field(..., description="Call to action text")


class ImageControlLevel1(BaseModel):
    enabled: bool = Field(..., description="Whether global image control is enabled")
    style: str = Field(..., description="Global image style preference")
    guidance: str = Field(..., description="Global image guidance instructions")
    caption: str = Field(..., description="Global image caption")
    ratio: str = Field(..., description="Global image aspect ratio")
    starting_image_url: Optional[str] = Field(None, description="URL of starting image for global control")


class PlatformImageControl(BaseModel):
    enabled: bool = Field(..., description="Whether platform-specific image control is enabled")
    style: str = Field(..., description="Platform-specific image style")
    guidance: str = Field(..., description="Platform-specific image guidance")
    caption: str = Field(..., description="Platform-specific image caption")
    ratio: str = Field(..., description="Platform-specific image aspect ratio")
    starting_image_url: Optional[str] = Field(None, description="URL of starting image for platform")


class ImageControlLevel2(BaseModel):
    facebook: Optional[PlatformImageControl] = None
    instagram: Optional[PlatformImageControl] = None
    linkedin: Optional[PlatformImageControl] = None
    twitter: Optional[PlatformImageControl] = None


class ImageControl(BaseModel):
    level_1: ImageControlLevel1 = Field(..., description="Global image control settings")
    level_2: ImageControlLevel2 = Field(..., description="Platform-specific image control overrides")


class Platform(BaseModel):
    platform: str = Field(..., description="Platform name (facebook|instagram|linkedin|twitter)")
    post_type: str = Field(..., description="Type of post for this platform")
    selected: bool = Field(..., description="Whether this platform is selected for generation")


class ContentGeneratorData(BaseModel):
    company: Company = Field(..., description="Company information")
    content: Content = Field(..., description="Content details")
    image_control: ImageControl = Field(..., description="Image generation control settings")
    platforms: List[Platform] = Field(..., description="Platform configurations")

    # Optional: Keep compatibility fields for transition
    language: str = Field(default="English", description="Target language for content")
    upload_to_cloud: bool = Field(default=True, description="Whether to upload to cloud storage")
    image_generation_model: Optional[str] = Field(
        default=None,
        description="Optional: Specify image generation model (e.g., 'imagen-4.0-fast-generate-001'). If not provided, uses default from config."
    )


class PipelineRequest(BaseModel):
    company_name: str = Field(
        ...,
        description="The name of the company for which content is being generated.",
        json_schema_extra={"example": "Creators Multiverse"}
    )
    company_mission: str = Field(
        ...,
        description="The mission statement of the company.",
        json_schema_extra={"example": "Empowering creators to build their digital presence with AI-powered tools..."}
    )
    company_sentiment: str = Field(
        ...,
        description="The desired sentiment and thematic elements for the company's voice.",
        json_schema_extra={"example": "Inspirational & Empowering. Cosmic/Magical Theme yet not too much."}
    )
    language: str = Field(
        default="English",
        description="The target language for the generated posts.",
        json_schema_extra={"example": "Spanish"}
    )
    platforms_post_types_map: List[Dict[str, str]] = Field(
        ...,
        description="A list of dictionaries specifying target platforms and their desired post types (e.g., Text, Image, Video).",
        json_schema_extra={"example": [{"linkedin": "Image"}, {"twitter": "Text"}]}
    )
    subject: str = Field(
        ...,
        description="The main subject or topic for the social media posts.",
        json_schema_extra={"example": "Hello World! Intro post about our company"}
    )
    tone: str = Field(
        default="Neutral",
        description="The desired tone for the generated posts.",
        json_schema_extra={"example": "Excited and Optimistic"}
    )
    requirements: Optional[List[RequirementItem]] = Field(
        default=None,
        description="Specific requirements or guidelines for the content generation."
    )
    posts_history: Optional[List[PostHistoryItem]] = Field(
        default=None,
        description="A list of previously generated posts for context."
    )
    upload_to_cloud: bool = Field(
        default=True,
        description="Flag to indicate whether generated assets should be uploaded to cloud storage."
    )
    image_generation_model: Optional[str] = Field(
        default=None,
        description="Optional: Specify image generation model (e.g., 'imagen-4.0-fast-generate-001'). If not provided, uses default from config.",
        json_schema_extra={"example": "imagen-4.0-fast-generate-001"}
    )

    # You can also provide an example for the whole model
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "company_name": "Tech Innovators Inc.",
                    "company_mission": "To boldly innovate where no tech has innovated before.",
                    "company_sentiment": "Futuristic and bold, slightly playful.",
                    "language": "English",
                    "platforms_post_types_map": [
                        {"linkedin": "Image"},
                        {"twitter": "Text"},
                        {"instagram": "Image"}
                    ],
                    "subject": "Announcing our new Quantum Entanglement Communicator!",
                    "tone": "Excited and Awe-Inspiring",
                    "requirements": [
                        {"type": "Call to Action", "detail": "Encourage users to visit our website for a demo."}
                    ],
                    "posts_history": [
                        {"platform": "twitter", "text": "Last week's #TechTuesday was a blast!"}
                    ],
                    "upload_to_cloud": True
                }
            ]
        }
    }


# You can also define a Pydantic model for the response if you want strict validation
# but returning the dict from the pipeline is often fine for internal APIs.
# class PipelineResponse(BaseModel):
#     pipeline_id: str
#     subject: str
#     # ... other fields from your summary
