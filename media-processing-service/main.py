"""
Media Processing Service
========================
Cloud Run service for video/image processing operations.
Uses FFmpeg for lossless video operations.

Endpoints:
- POST /video/export - Trim and concatenate video clips (lossless)
- GET /health - Health check
"""

import os
import asyncio
import subprocess
import uuid
import shutil
import aiohttp
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# ============================================
# Configuration
# ============================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
WORK_DIR = Path("/tmp/media-processing")

# ============================================
# FastAPI App Setup
# ============================================

app = FastAPI(
    title="Media Processing Service",
    description="Video and image processing API with FFmpeg",
    version="1.0.0"
)

# CORS - allow all origins for now (can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Models
# ============================================

class VideoClip(BaseModel):
    id: str
    sourceUrl: str
    sourceDuration: float
    startTime: float
    trimStart: float
    trimEnd: float


class VideoExportRequest(BaseModel):
    clips: List[VideoClip]
    userId: str
    companyId: Optional[str] = None
    projectName: Optional[str] = "Exported Video"


class VideoExportResponse(BaseModel):
    success: bool
    videoUrl: Optional[str] = None
    storagePath: Optional[str] = None
    fileSize: Optional[int] = None
    mediaFileId: Optional[str] = None
    processingTimeMs: Optional[int] = None
    error: Optional[str] = None


# ============================================
# Helper Functions
# ============================================

def get_supabase_client() -> Client:
    """Create Supabase client with service role key."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(
            status_code=500,
            detail="Supabase credentials not configured"
        )
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


async def download_file(url: str, dest_path: Path) -> None:
    """Download a file from URL to local path."""
    print(f"[Download] {url} -> {dest_path}")

    async with aiohttp.ClientSession() as session:
        async with session.get(url, allow_redirects=True) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download video: HTTP {response.status}"
                )

            async with aiofiles.open(dest_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    await f.write(chunk)

    print(f"[Download] Complete: {dest_path.stat().st_size} bytes")


def run_ffmpeg(args: List[str]) -> None:
    """Run FFmpeg command and handle errors."""
    cmd = ["ffmpeg", "-y"] + args
    print(f"[FFmpeg] Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"[FFmpeg] Error: {result.stderr}")
        raise HTTPException(
            status_code=500,
            detail=f"FFmpeg error: {result.stderr[:500]}"
        )

    print("[FFmpeg] Command completed successfully")


def trim_video(input_path: Path, output_path: Path, start_time: float, duration: float) -> None:
    """
    Trim video using FFmpeg with stream copy (lossless).

    Uses -ss before -i for fast seeking, -c copy to avoid re-encoding.
    """
    run_ffmpeg([
        "-ss", str(start_time),
        "-i", str(input_path),
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output_path)
    ])


def concatenate_videos(input_paths: List[Path], output_path: Path, work_dir: Path) -> None:
    """
    Concatenate videos using FFmpeg concat demuxer (lossless for same-codec files).
    """
    if len(input_paths) == 1:
        # Single file - just copy
        shutil.copy(input_paths[0], output_path)
        return

    # Create concat list file
    concat_list_path = work_dir / "concat_list.txt"
    with open(concat_list_path, "w") as f:
        for path in input_paths:
            f.write(f"file '{path}'\n")

    run_ffmpeg([
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_path),
        "-c", "copy",
        str(output_path)
    ])


# ============================================
# Endpoints
# ============================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Verify FFmpeg is available
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True
        )
        ffmpeg_available = result.returncode == 0
    except FileNotFoundError:
        ffmpeg_available = False

    return {
        "status": "healthy",
        "ffmpeg": ffmpeg_available,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/video/export", response_model=VideoExportResponse)
async def export_video(request: VideoExportRequest):
    """
    Export video by trimming and concatenating clips.

    Uses lossless stream copy (-c copy) to preserve original quality.
    Uploads result to Supabase storage and creates media_files record.
    """
    start_time = datetime.now()
    job_id = str(uuid.uuid4())[:8]
    work_dir = WORK_DIR / job_id

    print(f"[Export:{job_id}] Starting export with {len(request.clips)} clips")

    try:
        # Create work directory
        work_dir.mkdir(parents=True, exist_ok=True)

        # Sort clips by timeline position
        sorted_clips = sorted(request.clips, key=lambda c: c.startTime)

        # Step 1: Download all videos
        print(f"[Export:{job_id}] Step 1: Downloading videos...")
        downloaded_paths: List[Path] = []

        for i, clip in enumerate(sorted_clips):
            input_path = work_dir / f"input_{i}.mp4"
            await download_file(clip.sourceUrl, input_path)
            downloaded_paths.append(input_path)

        # Step 2: Trim each video (if needed)
        print(f"[Export:{job_id}] Step 2: Trimming videos...")
        trimmed_paths: List[Path] = []

        for i, clip in enumerate(sorted_clips):
            input_path = downloaded_paths[i]
            effective_duration = clip.sourceDuration - clip.trimStart - clip.trimEnd

            if clip.trimStart > 0 or clip.trimEnd > 0:
                # Need to trim
                trimmed_path = work_dir / f"trimmed_{i}.mp4"
                print(f"[Export:{job_id}] Trimming clip {i+1}: start={clip.trimStart}, duration={effective_duration}")
                trim_video(input_path, trimmed_path, clip.trimStart, effective_duration)
                trimmed_paths.append(trimmed_path)
            else:
                # No trim needed
                trimmed_paths.append(input_path)

        # Step 3: Concatenate all videos
        print(f"[Export:{job_id}] Step 3: Concatenating videos...")
        output_path = work_dir / "output.mp4"
        concatenate_videos(trimmed_paths, output_path, work_dir)

        # Get output file size
        output_size = output_path.stat().st_size
        print(f"[Export:{job_id}] Output file size: {output_size} bytes")

        # Step 4: Upload to Supabase storage
        print(f"[Export:{job_id}] Step 4: Uploading to storage...")
        supabase = get_supabase_client()

        timestamp = int(datetime.now().timestamp() * 1000)
        storage_path = f"{request.userId}/{request.companyId or 'default'}/{timestamp}_export.mp4"

        with open(output_path, "rb") as f:
            output_data = f.read()

        upload_result = supabase.storage.from_("media-studio-videos").upload(
            storage_path,
            output_data,
            file_options={"content-type": "video/mp4"}
        )

        # Get public URL
        public_url = supabase.storage.from_("media-studio-videos").get_public_url(storage_path)
        print(f"[Export:{job_id}] Uploaded to: {public_url}")

        # Step 5: Create media_files record
        print(f"[Export:{job_id}] Step 5: Creating media record...")
        safe_name = "".join(c for c in (request.projectName or "Exported Video") if c.isalnum() or c in " -_")

        # Calculate total duration
        total_duration = sum(
            clip.sourceDuration - clip.trimStart - clip.trimEnd
            for clip in sorted_clips
        )

        media_record = {
            "user_id": request.userId,
            "company_id": request.companyId if request.companyId else None,
            "file_name": f"{safe_name}.mp4",
            "file_type": "video",
            "file_format": "mp4",
            "file_size": output_size,
            "storage_path": storage_path,
            "public_url": public_url,
            "duration": int(total_duration),
            "prompt": f"Edited video: {safe_name}",
            "model_used": "editor-export",
        }

        result = supabase.table("media_files").insert(media_record).execute()
        media_file_id = result.data[0]["id"] if result.data else None

        # Calculate processing time
        processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        print(f"[Export:{job_id}] Complete in {processing_time_ms}ms")

        return VideoExportResponse(
            success=True,
            videoUrl=public_url,
            storagePath=storage_path,
            fileSize=output_size,
            mediaFileId=media_file_id,
            processingTimeMs=processing_time_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Export:{job_id}] Error: {str(e)}")
        return VideoExportResponse(
            success=False,
            error=str(e)
        )
    finally:
        # Cleanup work directory
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
            print(f"[Export:{job_id}] Cleaned up work directory")


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
