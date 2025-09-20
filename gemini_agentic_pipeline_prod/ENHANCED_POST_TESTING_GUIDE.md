# Enhanced Post API Testing Guide

This document provides a comprehensive guide on how to test the new "enhanced post" feature of the agentic pipeline.

## 1. Prerequisites

Before you begin, ensure you have the following tools installed:

*   **Python 3.8+:** The application is written in Python.
*   **Docker:** (Optional) For running the application in a containerized environment.
*   **Postman:** For sending API requests to the application.

## 2. Running the Pipeline

Follow these steps to run the agentic pipeline locally:

### 2.1. Install Dependencies

1.  Navigate to the `gemini_agentic_pipeline_prod` directory in your terminal.
2.  Install the required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

### 2.2. Start the Application

1.  Once the dependencies are installed, run the following command to start the FastAPI application:

    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```

2.  The application should now be running at `http://localhost:8000`.

## 3. API Endpoint Details

The enhanced post feature is exposed through the following API endpoint:

*   **URL:** `http://localhost:8000/generate-posts-enhanced`
*   **HTTP Method:** `POST`
*   **Headers:**
    *   `Content-Type`: `application/json`

### 3.1. Request Body

The request body must be a JSON object with the following structure. See `test_enhanced_api.py` for a detailed example.

```json
{
    "company": {
        "id": "test-company-123",
        "name": "Creators Multiverse",
        "mission": "Empowering creators to build their digital presence with AI-powered tools",
        "tone_of_voice": "Inspirational & Empowering with a Cosmic/Magical theme",
        "primary_color_1": "#6366f1",
        "primary_color_2": "#ec4899",
        "logo_path": null
    },
    "content": {
        "topic": "Introducing our revolutionary AI content creation platform",
        "description": "Launch announcement for our new AI-powered social media content generation tool that helps creators build amazing content across platforms",
        "hashtags": ["AI", "ContentCreation", "SocialMedia", "CreatorsMultiverse"],
        "call_to_action": "Visit our website to start creating amazing content today!"
    },
    "image_control": {
        "level_1": {
            "enabled": true,
            "style": "Modern, vibrant, tech-inspired digital art",
            "guidance": "Focus on creativity and innovation, use cosmic/magical elements subtly",
            "caption": "AI-powered content creation revolution",
            "ratio": "1:1",
            "starting_image_url": null
        },
        "level_2": {
            "instagram": {
                "enabled": true,
                "style": "Instagram-optimized, colorful and engaging",
                "guidance": "Perfect for Instagram feed, make it pop with vibrant colors",
                "caption": "Perfect for your Instagram feed",
                "ratio": "1:1",
                "starting_image_url": null
            },
            "linkedin": {
                "enabled": true,
                "style": "Professional, clean, business-oriented",
                "guidance": "Corporate and professional look suitable for LinkedIn",
                "caption": "Professional announcement",
                "ratio": "1.91:1",
                "starting_image_url": null
            }
        }
    },
    "platforms": [
        {"platform": "instagram", "post_type": "Image", "selected": true},
        {"platform": "linkedin", "post_type": "Image", "selected": true},
        {"platform": "twitter", "post_type": "Text", "selected": true},
        {"platform": "facebook", "post_type": "Image", "selected": false}
    ],
    "language": "English",
    "upload_to_cloud": false
}
```

## 4. Testing with Postman

1.  **Create a new request:** Open Postman and create a new `POST` request.
2.  **Set the URL:** Enter `http://localhost:8000/generate-posts-enhanced` as the request URL.
3.  **Set the headers:** Go to the "Headers" tab and add a new header with `Content-Type` as the key and `application/json` as the value.
4.  **Set the body:**
    *   Go to the "Body" tab.
    *   Select the "raw" radio button.
    *   Choose "JSON" from the dropdown menu.
    *   Paste the JSON request body from section 3.1 into the text area.
5.  **Send the request:** Click the "Send" button to send the request to the application.

## 5. Expected Outcome

A successful request will return a `200 OK` status code and a JSON response body similar to the following:

```json
{
    "pipeline_id": "some-unique-id",
    "posts": [
        {
            "platform": "instagram",
            "post_type": "Image",
            "original_text_content": "...",
            "media_asset": {
                "file_path": "...",
                "url": "..."
            },
            "media_generation_prompt_used": "..."
        },
        {
            "platform": "linkedin",
            "post_type": "Image",
            "original_text_content": "...",
            "media_asset": {
                "file_path": "...",
                "url": "..."
            },
            "media_generation_prompt_used": "..."
        },
        {
            "platform": "twitter",
            "post_type": "Text",
            "original_text_content": "..."
        }
    ],
    "image_controls_used": {
        "level_1_enabled": true,
        "level_2_overrides": ["instagram", "linkedin"]
    }
}
```

## 6. Test Cases

Here are a few test cases to consider:

*   **Valid Request:** Use the request body from section 3.1. The pipeline should execute successfully.
*   **Invalid Request:** Send a request with a missing required field (e.g., `company`). The API should return a `422 Unprocessable Entity` error.
*   **Cloud Upload:** Set `upload_to_cloud` to `true`. The pipeline should attempt to upload the generated media to the configured cloud storage. (Note: This may require additional configuration).
*   **Different Platforms:** Modify the `platforms` array to select different platforms and post types. The pipeline should generate content for the selected platforms.
*   **Disable Image Control:** Set `image_control.level_1.enabled` to `false`. The pipeline should not generate any images.
