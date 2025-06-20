# data_models.py
from typing import Dict, List, TypedDict, Optional, Literal, Union, Any


class PostHistoryEntry(TypedDict):
    post_type: Literal["A", "B", "C"]  # Future: "D", "E" for video
    count: int
    score: int


class Requirements(TypedDict):
    min_length: Optional[int]
    max_length: Optional[int]
    must_include_keywords: Optional[List[str]]


class Layer2Input(TypedDict):
    company_name: str
    company_mission: str
    company_sentiment: str
    subject: str
    platforms_to_target: List[str]
    requirements: Optional[Requirements]
    posts_history: Optional[List[PostHistoryEntry]]
    language: str
    tone: str
    platform_post_type: str


class Layer2Output(TypedDict):
    core_post_text: str


class PlatformAgentInput(TypedDict):
    company_name: str
    company_mission: str
    company_sentiment: str
    language: str  # New: To specify the language for this platform's post
    tone: str
    subject: str
    platform_post_type: str  # New: e.g., "Text", "Image", "Video" from PLATFORMS_POST_TYPES_MAP
    #    post_type_decision: Literal["A", "B", "C"]  # Future: "D", "E" # Correctly commented out/removed
    core_post_text_suggestion: str
    target_platform: str


class PlatformAgentOutput(TypedDict):
    platform_specific_text: str
    platform_media_generation_prompt: Optional[str]  # Prompt for image/video


class SavedMediaAsset(TypedDict):
    type: Literal["image", "video"]  # Currently only 'image'
    file_path: str  # Path to the locally saved file


# --- Cloud Storage Models ---
class CloudStorageInfo(TypedDict):
    """Information about a file stored in cloud storage."""
    success: bool
    cloud_path: Optional[str]
    public_url: Optional[str]
    bucket_name: Optional[str]
    content_type: Optional[str]
    size: Optional[int]
    uploaded_at: Optional[str]
    error: Optional[str]


class SavedMediaAsset(TypedDict):
    """Information about a saved media asset (local and cloud)."""
    type: str  # "image", "video", etc.
    file_path: str  # Local file path
    cloud_storage: Optional[CloudStorageInfo]  # Cloud storage information


class CloudUploadResult(TypedDict):
    """Result of uploading files for a platform to cloud storage."""
    platform: str
    filename_base: str
    uploads: List[CloudStorageInfo]  # List of upload results for each file


class FinalGeneratedPost(TypedDict):
    """Complete information about a generated social media post."""
    platform: str
    post_type: str
    text_file_path: str
    media_asset: Optional[SavedMediaAsset]
    original_text_content: str
    media_generation_prompt_used: Optional[str]
    cloud_storage: Optional[CloudUploadResult]  # Cloud storage information


# --- Pipeline Summary Models ---
class PipelineSummary(TypedDict):
    """Complete summary of a pipeline execution."""
    pipeline_id: str
    subject: str
    post_type: str
    platforms: List[str]
    generated_at: str  # ISO format timestamp
    posts: List[FinalGeneratedPost]
    cloud_uploads: Optional[List[Dict[str, Any]]]  # Per-platform upload results
    requirements: Optional[Requirements]
    posts_history: Optional[List[PostHistoryEntry]]
    summary_cloud_storage: Optional[CloudStorageInfo]


class TranslatorAgentInput(TypedDict):
    text_to_translate: str  # This will be the platform_specific_text from Layer 3
    target_language: str
    company_name: str  # For context
    company_mission: str  # For context
    original_subject: str  # For context (the initial subject given to Layer 1/2)


class TranslatorAgentOutput(TypedDict):
    translated_text: str
