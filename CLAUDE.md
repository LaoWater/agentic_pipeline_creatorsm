# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a dual-ecosystem agentic pipeline for social media content generation:

1. **agentic_pipeline_creatorsm_fullgemini/**: Uses Google's Gemini ecosystem (Google GenAI, Google Cloud Storage)
2. **agentic_pipeline_openai/**: Uses OpenAI ecosystem (OpenAI API, LangChain)
3. **Exploring_google_Imagen/**: Experimental scripts for Google Imagen integration

Both pipelines provide FastAPI-based REST APIs for generating social media posts with AI-driven content creation, visual asset generation, and multi-platform adaptation.

## Architecture

### Core Components
- **FastAPI Application** (`main.py`): REST API server with CORS support
- **Pipeline Orchestrator** (`pipeline_orchestrator.py`): Main workflow coordination
- **LLM Services** (`llm_services.py`): AI model interactions (Layer 2 decision maker, platform adaptation, translation)
- **Media Generation** (`media_generation.py`): Visual asset creation using AI models
- **Cloud Storage Service** (`cloud_storage_service.py`): File upload and management
- **Data Models** (`data_models.py`): Pydantic models for type safety
- **API Models** (`api_models.py`): Request/response schemas

### Pipeline Flow
1. **Layer 2 Decision Maker**: Analyzes subject and company context
2. **Platform Adaptation Agent**: Customizes content for each target platform
3. **Translation Agent**: Handles multi-language support
4. **Visual Asset Generation**: Creates platform-specific images
5. **Cloud Storage**: Uploads generated content and returns URLs

## Development Commands

### Running the Application
```bash
# For Gemini pipeline
cd agentic_pipeline_creatorsm_fullgemini
python main.py

# For OpenAI pipeline  
cd agentic_pipeline_openai
python main.py
```

### Docker Deployment
```bash
# Build Docker image
docker build -t agentic-pipeline .

# Run container
docker run -p 8000:8000 agentic-pipeline
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Required environment variables:
# - Google GenAI API key (for Gemini pipeline)
# - OpenAI API key (for OpenAI pipeline)
# - Google Cloud Storage credentials
```

## Key Technical Details

### API Endpoints
- **POST /generate-posts**: Original pipeline endpoint accepting `PipelineRequest`
- **POST /generate-posts-enhanced**: Enhanced pipeline with image controls accepting `ContentGeneratorData`
- Default port: 8000
- CORS enabled for localhost:8080, localhost:5173, and creators-multiverse.com

### Enhanced Image Control System
The new `/generate-posts-enhanced` endpoint supports hierarchical image control:

- **Level 1 (Global)**: Default image settings applied to all platforms
- **Level 2 (Platform-specific)**: Override settings for individual platforms
- **Level 2 always overrides Level 1** when both exist for a platform
- **Starting Images**: Download and use existing images as base for generation
- **Brand Colors**: Automatic injection of company primary/secondary colors
- **Style & Guidance Controls**: Custom styling and generation instructions
- **Aspect Ratio Control**: Platform-specific image dimensions

### Supported Platforms
- Instagram (stories, posts, reels)
- TikTok (videos, stories)  
- LinkedIn (posts, articles)
- Facebook (posts, stories)
- YouTube (shorts, thumbnails)

### File Structure
- Generated content saved to `BASE_OUTPUT_FOLDER` (configurable)
- Platform-specific subfolders created automatically
- Cloud storage uploads return public URLs

### Dependencies
- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation
- **google-genai**: Google AI models (Gemini pipeline)
- **openai + langchain**: OpenAI integration (OpenAI pipeline)
- **google-cloud-storage**: Cloud file storage
- **Pillow**: Image processing

## Ecosystem Selection Strategy

The repository implements a dual-system architecture to serve different user preferences:
- **Google Ecosystem**: More structured, enclosed approach suitable for businesses preferring Google's methodology
- **OpenAI Ecosystem**: More exploratory, creative approach for users who prefer OpenAI's style

Choose the appropriate pipeline based on company sentiment and content creation philosophy.