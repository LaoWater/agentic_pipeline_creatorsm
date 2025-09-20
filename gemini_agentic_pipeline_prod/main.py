# main.py
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import uvicorn

# Ensure your project structure allows these imports
# If pipeline_orchestrator is in the same directory:
from pipeline_orchestrator import generate_social_media_posts_pipeline, generate_enhanced_social_media_posts_pipeline
from api_models import PipelineRequest, ContentGeneratorData
from config import BASE_OUTPUT_FOLDER  # For ensuring output folder exists

app = FastAPI(
    title="CreatorsM Agentic Pipeline",
    description="API to generate social media posts.",
    version="1.0.0"
)

origins = [
    "http://localhost:8080",  # Your React dev server
    "http://localhost:5173",  # Vite dev server (common alternative)
    # Add your deployed frontend URL here when you deploy
    "https://creators-multiverse.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# Ensure the base output folder exists when the app starts
os.makedirs(BASE_OUTPUT_FOLDER, exist_ok=True)
print(f"Base output folder ensured at: {BASE_OUTPUT_FOLDER}")


@app.on_event("startup")
async def startup_event():
    # You can add any startup logic here, e.g., initializing LLM clients if not done at import
    # Or checking for necessary environment variables (API keys)
    print("FastAPI application started.")


async def run_pipeline_background(request_data: PipelineRequest):
    """Helper to run the pipeline, to be called by BackgroundTasks or directly."""
    try:
        print(f"Received pipeline request for company: {request_data.company_name}, subject: {request_data.subject}")
        # Convert Pydantic models to dicts if your pipeline expects plain dicts for requirements/history
        requirements_dict = [r.model_dump() for r in request_data.requirements] if request_data.requirements else None
        posts_history_dict = [p.model_dump() for p in
                              request_data.posts_history] if request_data.posts_history else None

        # Call the refactored pipeline function
        result = await generate_social_media_posts_pipeline(
            subject=request_data.subject,
            company_name=request_data.company_name,
            company_mission=request_data.company_mission,
            company_sentiment=request_data.company_sentiment,
            language=request_data.language,
            platforms_post_types_map=request_data.platforms_post_types_map,
            tone=request_data.tone,
            requirements=requirements_dict,  # Pass the converted dict
            posts_history=posts_history_dict,  # Pass the converted dict
            upload_to_cloud=request_data.upload_to_cloud
        )
        print(
            f"Pipeline completed for company: {request_data.company_name}, subject: {request_data.subject}. Result ID: {result.get('pipeline_id')}")
        # If running in background, you might store the result in a DB or notify via webhook.
        # For a direct response, this is fine.
        return result
    except Exception as e:
        print(
            f"Error processing pipeline for company: {request_data.company_name}, subject: {request_data.subject}. Error: {e}")
        # For a direct response, re-raise or return an error structure
        # If in background, log extensively.
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")


@app.post("/generate-posts", summary="Trigger Social Media Post Generation Pipeline")
async def create_social_media_posts(
        request: PipelineRequest,
        # background_tasks: BackgroundTasks # Option 1: Run in background
):
    """
    Accepts company and post details, generates content, and returns GCloud links.
    """
    # Option 1: Run in background and return immediately (e.g., with a task ID)
    # This is better for long-running tasks. The client would poll for status or receive a webhook.
    # task_id = "some_unique_task_id"
    # background_tasks.add_task(run_pipeline_background, request)
    # return {"message": "Pipeline processing started.", "task_id": task_id}

    # Option 2: Run synchronously (for the API call) and wait for completion
    # This is simpler if the pipeline is reasonably fast or if the client can wait.
    try:
        result = await run_pipeline_background(request)
        if result.get("error"):
            # If the pipeline itself returns a structured error
            return JSONResponse(status_code=400, content=result)
        return result
    except HTTPException as http_exc:  # Catch HTTPExceptions raised by run_pipeline_background
        raise http_exc
    except Exception as e:
        print(f"Unhandled error in /generate-posts endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/generate-posts-enhanced", summary="Enhanced Social Media Post Generation with Image Controls")
async def create_enhanced_social_media_posts(
        request: ContentGeneratorData,
        # background_tasks: BackgroundTasks # Option 1: Run in background
):
    """
    Enhanced endpoint that accepts the new ContentGeneratorData structure with image controls.
    Supports hierarchical image control (Level 1 global, Level 2 platform-specific).
    """
    try:
        result = await run_enhanced_pipeline_background(request)
        if result.get("error"):
            return JSONResponse(status_code=400, content=result)
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Unhandled error in /generate-posts-enhanced endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


async def run_enhanced_pipeline_background(request_data: ContentGeneratorData):
    """Helper to run the enhanced pipeline with new ContentGeneratorData structure."""
    try:
        print(f"Received enhanced pipeline request for company: {request_data.company.name}, topic: {request_data.content.topic}")
        
        # Call the enhanced pipeline function
        result = await generate_enhanced_social_media_posts_pipeline(request_data)
        
        print(f"Enhanced pipeline completed for company: {request_data.company.name}. Result ID: {result.get('pipeline_id')}")
        return result
        
    except Exception as e:
        print(f"Error processing enhanced pipeline for company: {request_data.company.name}. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced pipeline processing failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)  # reload=True for dev
