# CreatorsM Agentic Pipeline - Documentation

## Overview

This project provides an API for generating social media posts using an agentic pipeline. It leverages Google Cloud Storage for file management and the Google Imagen model for image generation. The API is built with FastAPI and allows users to specify the company, subject, and other parameters for generating customized social media content.

## Project Structure

- `main.py`:  The main FastAPI application, handling API requests and coordinating the pipeline.
- `pipeline_orchestrator.py`: (Not provided, but essential) This file should contain the core logic for orchestrating the social media post generation process, including interactions with language models, media generation, and cloud storage.
- `api_models.py`: (Not provided, but essential) Defines the Pydantic models used for request and response data validation.  Crucially, it should define the `PipelineRequest` model.
- `cloud_storage_service.py`:  Handles interactions with Google Cloud Storage, including uploading files and managing access control.
- `media_generation.py`:  Orchestrates the generation of visual assets (currently images) using the Google Imagen model.
- `visual_model.py`:  Contains the implementation for interacting with the Google Imagen API for image generation.
- `.env`: Stores configuration settings, including API keys and cloud storage details.  **Important: Do not commit this file to version control.**

## Getting Started

### Prerequisites

- Python 3.9+
- Google Cloud Project with the Imagen API enabled
- Google Cloud Storage bucket
- Service account with necessary permissions (Storage Object Admin for GCS)
- API keys for Google Imagen and (optionally) other services.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd v2_agentic_pipeline
    ```

2.  **Create a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt  # Assuming you have a requirements.txt
    ```
    **Note:** If a `requirements.txt` file is not present in the provided context, you can create one and populate it based on the imported packages in the code. For example:

    ```
    fastapi
    uvicorn[standard]
    google-cloud-storage
    google-cloud-aiplatform
    python-dotenv
    Pillow
    ```
    You may need to add or remove packages based on the contents of `pipeline_orchestrator.py` and `api_models.py`.

4.  **Set up environment variables:**

    - Create a `.env` file in the project root and add the following (replace with your actual values):

      ```
      GCS_PROJECT_ID=<your-gcp-project-id>
      GCS_BUCKET_NAME=<your-gcs-bucket-name>
      GCS_SERVICE_ACCOUNT_KEY_PATH=<path-to-your-service-account-key-json>
      GEMINI_API_KEY=<your-gemini-api-key>
      # OPENAI_API_KEY_VALUE=<your-openai-api-key>  # If using OpenAI models
      ```

    - **Security:** Never commit your `.env` file to version control. Add it to your `.gitignore`.

5.  **Configure CORS (in `main.py`):**

    -  Update the `origins` list in the `CORSMiddleware` configuration to include the URL of your frontend application(s).

### Running the API

1.  **Start the server:**

    ```bash
    python main.py
    ```

    Alternatively, use uvicorn directly:

    ```bash
    uvicorn main:app --reload  # For development with automatic reloading
    ```

    The API will typically be available at `http://127.0.0.1:8000`.

## API Usage

### `POST /generate-posts`

-   **Summary:** Triggers the social media post generation pipeline.
-   **Request Body:**  JSON data conforming to the `PipelineRequest` model (defined in `api_models.py`). You'll need to inspect that file to see the exact structure, but it will likely include fields like:

    ```json
    {
      "company_name": "Acme Corp",
      "subject": "New Product Launch",
      "company_mission": "To provide innovative solutions...",
      "company_sentiment": "positive",
      "language": "en",
      "platforms_post_types_map": {
        "facebook": ["image", "text"],
        "twitter": ["text"]
      },
      "tone": "Excited and informative",
      "requirements": [
        {"type": "length", "max_words": 150},
        {"type": "keywords", "include": ["innovation", "new product"]}
      ],
      "posts_history": [
        {"platform": "facebook", "content": "Previous post content...", "date": "2023-10-26"}
      ],
      "upload_to_cloud": true
    }
    ```

-   **Response:**  A JSON object containing:

    -   `pipeline_id`: A unique identifier for the pipeline execution.
    -   `results`: Details about the generated posts, including:
        -   `platform`:  The target platform.
        -   `filename_base`:  Base filename for generated files.
        -   `uploads`:  A list of upload results (from `cloud_storage_service.py`), including:
            -   `success`:  Boolean indicating successful upload.
            -   `cloud_path`: Path to the file in Google Cloud Storage.
            -   `public_url`: Public URL of the uploaded file.
            -   Other metadata.

    -  If an error occurs, the response will include an `error` field with details.

-   **Example Request (using `curl`):**

    ```bash
    curl -X POST -H "Content-Type: application/json" \
      -d '{"company_name": "Example Co", "subject": "Sample Post", ...}' \
      http://127.0.0.1:8000/generate-posts
    ```

## Google Cloud Storage

The `cloud_storage_service.py` module handles uploading generated content to Google Cloud Storage.  Files are organized within the bucket by timestamp, platform, and type (text/image).

-   **Permissions:** Ensure the service account specified in `GCS_SERVICE_ACCOUNT_KEY_PATH` has the "Storage Object Admin" role on the target bucket.
-   **Public Access:** Uploaded files are made publicly readable by default. Review and adjust the ACL settings in `cloud_storage_service.py` if you need different access control.

## Image Generation with Google Imagen

The `media_generation.py` and `visual_model.py` modules handle image generation using the Google Imagen API.

-   **Authentication:**  The `GEMINI_API_KEY` environment variable must be set with a valid API key.
-   **Model Selection:** The code currently defaults to `imagen-3.0-generate-002`. You can change this in `media_generation.py` or provide a `model` parameter per task. Consult the Google documentation for the latest available models and their capabilities.

## Error Handling

The API implements basic error handling.  Errors during pipeline execution or file uploads will result in HTTP 500 errors with details in the response body.  The code also includes logging to help with debugging.

## Further Development

-   **Pipeline Orchestration:**  The `pipeline_orchestrator.py` needs to be fully implemented to integrate language models for content generation, manage the workflow, and handle interactions between different components.
-   **API Models:** Define the `PipelineRequest` model and any other necessary models in `api_models.py` to ensure proper request validation and data handling.
-   **More Media Types:** Extend `media_generation.py` to support video generation or other media types.
-   **Advanced Error Handling:** Implement more robust error handling, including specific error codes and user-friendly messages.
-   **Testing:**  Write unit and integration tests to ensure code quality and reliability.
-   **Deployment:**  Deploy the API to a platform like Google Cloud Run or similar for production use.

## Contributing

Contributions are welcome!  Please follow the standard Git workflow: fork the repository, create a branch for your changes, and submit a pull request.