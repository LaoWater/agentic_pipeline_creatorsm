# cloud_storage_service.py
# ACL -
import os
import json
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
import mimetypes
from dotenv import load_dotenv, dotenv_values

load_dotenv()  # Load environment variables from .env file

config = dotenv_values(".env")


def get_env_or_config(var_name: str, is_required: bool = True, default_value: Optional[str] = None) -> Optional[str]:
    """
    Retrieves a configuration value.
    Priority:
    1. Environment variable (os.getenv)
    2. .env file (via load_dotenv() already populating os.getenv, or directly from dotenv_values if you prefer)
    3. Default value
    """
    value = os.getenv(var_name) # load_dotenv() makes .env values available here if not already in env

    if value is not None:
        print(f"[INFO] Using {var_name} from environment or .env file: '{value}'") # Be careful logging sensitive values
    elif default_value is not None:
        value = default_value
        print(f"[INFO] Using default value for {var_name}: '{value}'")
    elif is_required:
        print(f"[ERROR] Missing required config: {var_name}. Set it in environment or .env file.")
        raise ValueError(f"Missing required config: {var_name}. Set it in environment or .env file.")
    else:
        print(f"[WARN] Optional config {var_name} not set, using None.")
        return None # Return None for optional missing values, not ""

    return value

# Configuration from environment variables or .env file
# These are set by Cloud Run environment variables
GCS_BUCKET_NAME = get_env_or_config("GCS_BUCKET_NAME", is_required=True)
GCS_PROJECT_ID = get_env_or_config("GCS_PROJECT_ID", is_required=True)

# This is OPTIONAL and should NOT be required for Cloud Run.
# It will be None in Cloud Run, and your _get_client() handles that.
GCS_SERVICE_ACCOUNT_KEY_PATH = get_env_or_config("GCS_SERVICE_ACCOUNT_KEY_PATH", is_required=False)

print(f"Global Config Loaded: GCS_BUCKET_NAME='{GCS_BUCKET_NAME}', GCS_PROJECT_ID='{GCS_PROJECT_ID}', GCS_SERVICE_ACCOUNT_KEY_PATH='{GCS_SERVICE_ACCOUNT_KEY_PATH}'")



class CloudStorageService:
    """Service for uploading and managing files in Google Cloud Storage."""

    def __init__(self, bucket_name: str = GCS_BUCKET_NAME, project_id: str = GCS_PROJECT_ID):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self._client = None
        self._bucket = None

    def _get_client(self) -> storage.Client:
        """Get or create the GCS client, handling authentication."""
        if self._client is None:
            try:
                if GCS_SERVICE_ACCOUNT_KEY_PATH and os.path.exists(GCS_SERVICE_ACCOUNT_KEY_PATH):
                    self._client = storage.Client.from_service_account_json(
                        GCS_SERVICE_ACCOUNT_KEY_PATH,
                        project=self.project_id
                    )
                else:
                    self._client = storage.Client(project=self.project_id)
            except Exception as e:
                raise Exception(f"Failed to initialize GCS client: {e}")
        print("Initializing GCS client with Service Account: " + GCS_SERVICE_ACCOUNT_KEY_PATH)
        return self._client

    def _get_bucket(self) -> storage.Bucket:
        """Get or create the GCS bucket reference."""
        if self._bucket is None:
            client = self._get_client()
            self._bucket = client.bucket(self.bucket_name)
        return self._bucket

    async def upload_file(
            self,
            local_file_path: str,
            cloud_file_path: str,
            content_type: Optional[str] = None,
            metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Uploads a file and makes it publicly readable."""
        try:
            if not os.path.exists(local_file_path):
                raise FileNotFoundError(f"Local file not found: {local_file_path}")

            bucket = self._get_bucket()
            blob = bucket.blob(cloud_file_path)

            if content_type is None:
                content_type, _ = mimetypes.guess_type(local_file_path)
                content_type = content_type or 'application/octet-stream'

            if metadata:
                blob.metadata = metadata

            # Upload the file from a path and make it public
            with open(local_file_path, 'rb') as file_data:
                blob.upload_from_file(
                    file_data,
                    content_type=content_type,
                    predefined_acl='public-read'  # <-- THE FIX
                )

            public_url = blob.public_url
            upload_result = {
                "success": True, "cloud_path": cloud_file_path, "public_url": public_url,
                "bucket_name": self.bucket_name, "content_type": content_type,
                "size": os.path.getsize(local_file_path), "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "local_path": local_file_path
            }
            print(f"✅ Publicly uploaded {local_file_path} to {cloud_file_path}")
            return upload_result
        except Exception as e:
            error_msg = f"Upload failed for {local_file_path}: {e}"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg, "local_path": local_file_path, "cloud_path": cloud_file_path}

    async def upload_text_content(
            self,
            text_content: str,
            cloud_file_path: str,
            metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Uploads text content and makes it publicly readable."""
        try:
            bucket = self._get_bucket()
            blob = bucket.blob(cloud_file_path)

            if metadata:
                blob.metadata = metadata

            # Upload text from a string and make it public
            blob.upload_from_string(
                text_content,
                content_type='text/plain; charset=utf-8',
                predefined_acl='public-read'  # <-- THE FIX
            )

            public_url = blob.public_url
            upload_result = {
                "success": True, "cloud_path": cloud_file_path, "public_url": public_url,
                "bucket_name": self.bucket_name, "content_type": "text/plain; charset=utf-8",
                "size": len(text_content.encode('utf-8')), "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "content_preview": text_content[:100] + "..." if len(text_content) > 100 else text_content
            }
            print(f"✅ Publicly uploaded text content to {cloud_file_path}")
            return upload_result
        except Exception as e:
            error_msg = f"Text upload failed for {cloud_file_path}: {e}"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg, "cloud_path": cloud_file_path}

    async def upload_json_data(
            self,
            json_data: Dict[Any, Any],
            cloud_file_path: str,
            metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Uploads JSON data and makes it publicly readable."""
        try:
            json_content = json.dumps(json_data, indent=2, ensure_ascii=False)
            bucket = self._get_bucket()
            blob = bucket.blob(cloud_file_path)

            if metadata:
                blob.metadata = metadata

            # Upload JSON from a string and make it public
            blob.upload_from_string(
                json_content,
                content_type='application/json; charset=utf-8',
                predefined_acl='public-read'  # <-- THE FIX
            )

            public_url = blob.public_url
            upload_result = {
                "success": True, "cloud_path": cloud_file_path, "public_url": public_url,
                "bucket_name": self.bucket_name, "content_type": "application/json; charset=utf-8",
                "size": len(json_content.encode('utf-8')), "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "record_count": len(json_data) if isinstance(json_data, (list, dict)) else 1
            }
            print(f"✅ Publicly uploaded JSON data to {cloud_file_path}")
            return upload_result
        except Exception as e:
            error_msg = f"JSON upload failed for {cloud_file_path}: {e}"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg, "cloud_path": cloud_file_path}

    def generate_cloud_path(self, filename_base: str, platform: str, file_type: str, extension: str) -> str:
        """Generates a standardized cloud storage path."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"social_media_posts/{timestamp}/{platform}/{file_type}/{filename_base}.{extension}"


# Initialize a global instance for easy importing
cloud_storage = CloudStorageService()


# --- Utility Functions (no changes needed here) ---
async def upload_generated_post_files(
        filename_base: str,
        platform: str,
        text_content: str,
        media_file_path: Optional[str] = None,
        media_generation_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Upload all files related to a generated social media post."""
    upload_results = {
        "platform": platform,
        "filename_base": filename_base,
        "uploads": []
    }

    # Upload text content
    text_cloud_path = cloud_storage.generate_cloud_path(filename_base, platform, "text", "txt")
    text_metadata = {"platform": platform, "content_type": "social_media_text", "filename_base": filename_base}
    text_result = await cloud_storage.upload_text_content(text_content, text_cloud_path, text_metadata)
    upload_results["uploads"].append(text_result)

    # Upload media file if provided
    if media_file_path and os.path.exists(media_file_path):
        _, ext = os.path.splitext(media_file_path)
        ext = ext.lstrip('.')
        media_cloud_path = cloud_storage.generate_cloud_path(filename_base, platform, "image", ext)
        media_metadata = {
            "platform": platform, "content_type": "social_media_image", "filename_base": filename_base,
            "generation_prompt": media_generation_prompt[:500] if media_generation_prompt else ""
        }
        media_result = await cloud_storage.upload_file(media_file_path, media_cloud_path, metadata=media_metadata)
        upload_results["uploads"].append(media_result)

    return upload_results
