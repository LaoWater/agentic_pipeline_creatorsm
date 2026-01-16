# Media Processing Service

A dedicated Cloud Run service for video and image processing operations, designed to work with the BrandVerse AI Studio video editor.

## Features

- **Lossless Video Export**: Trim and concatenate video clips using FFmpeg's stream copy mode (`-c copy`) - preserves original quality with no re-encoding
- **Server-Side Processing**: No browser limitations (no SharedArrayBuffer/COOP/COEP issues)
- **Auto-Save to Library**: Exported videos are automatically saved to Supabase storage and added to the user's media library

## Architecture

```
┌─────────────────────────┐     ┌──────────────────────────┐
│  BrandVerse Frontend    │────▶│  Media Processing Svc    │
│  (Video Editor)         │     │  (Cloud Run + FFmpeg)    │
└─────────────────────────┘     └──────────────────────────┘
                                           │
                                           ▼
                               ┌──────────────────────────┐
                               │  Supabase Storage        │
                               │  (media-studio-videos)   │
                               └──────────────────────────┘
```

## API Endpoints

### `POST /video/export`

Export video by trimming and concatenating clips.

**Request:**
```json
{
  "clips": [
    {
      "id": "clip1",
      "sourceUrl": "https://...",
      "sourceDuration": 10.0,
      "startTime": 0,
      "trimStart": 2.0,
      "trimEnd": 1.0
    }
  ],
  "userId": "user-uuid",
  "companyId": "company-uuid",
  "projectName": "My Video"
}
```

**Response:**
```json
{
  "success": true,
  "videoUrl": "https://...",
  "storagePath": "user-uuid/company-uuid/1234567890_export.mp4",
  "fileSize": 1234567,
  "mediaFileId": "media-file-uuid",
  "processingTimeMs": 5432
}
```

### `GET /health`

Health check endpoint.

## Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for full deployment instructions.

**Quick deploy:**
```bash
./deploy.sh
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key |
| `PORT` | Server port (default: 8080) |

## Future Endpoints

This service is designed to expand with:

- `POST /video/effects` - Apply effects, transitions
- `POST /image/edit` - Image cropping, filters, overlays
- `POST /audio/process` - Audio extraction, mixing
- `POST /video/upscale` - AI-powered video upscaling (GPU instances)

## Tech Stack

- **Python 3.12** + **FastAPI**
- **FFmpeg** for video processing
- **Supabase** for storage and database
- **Google Cloud Run** for hosting
