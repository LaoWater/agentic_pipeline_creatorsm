# api_models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


# Re-use from your existing data_models if they fit, or define specific for API
class RequirementItem(BaseModel):  # Example, adapt from your Requirements TypedDict
    type: str
    detail: str


class PostHistoryItem(BaseModel):  # Example, adapt from your PostHistoryEntry
    platform: str
    text: str
    # ... other fields


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
