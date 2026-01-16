# Media Processing Service - Deployment Guide

## Overview
Cloud Run service for video/image processing with FFmpeg.
Provides lossless video trimming and concatenation for the video editor.

## Prerequisites
- Google Cloud SDK installed and configured
- Project: `creators-multi-verse`
- Supabase credentials stored in Secret Manager

---

## First-Time Setup (One Time Only)

### 1. Store Supabase credentials in Secret Manager

```bash
# Create secrets (run once)
echo -n "YOUR_SUPABASE_URL" | gcloud secrets create SUPABASE_URL --data-file=- --project=creators-multi-verse
echo -n "YOUR_SUPABASE_SERVICE_ROLE_KEY" | gcloud secrets create SUPABASE_SERVICE_ROLE_KEY --data-file=- --project=creators-multi-verse
```

### 2. Grant Cloud Run access to secrets

```bash
# Get the compute service account
gcloud iam service-accounts list --project=creators-multi-verse

# Grant secret accessor role (replace with your service account)
gcloud secrets add-iam-policy-binding SUPABASE_URL \
  --member="serviceAccount:YOUR_COMPUTE_SA@creators-multi-verse.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" \
  --project=creators-multi-verse

gcloud secrets add-iam-policy-binding SUPABASE_SERVICE_ROLE_KEY \
  --member="serviceAccount:YOUR_COMPUTE_SA@creators-multi-verse.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" \
  --project=creators-multi-verse
```

---

## Deployment (MAC)

Open terminal in this folder (`media-processing-service`) and run:

### 1. Define Variables

```bash
# --- Define variables ---
PROJECT_ID="creators-multi-verse"
REGION="europe-central2"
AR_REPO="creatorsm-docker-repo"
IMAGE_NAME="media-processing-service"
IMAGE_TAG=$(date +%Y%m%d-%H%M%S)
AR_IMAGE_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"
SERVICE_NAME="media-processing-svc"

# Verify
echo "Deploying: $IMAGE_NAME:$IMAGE_TAG"
echo "Service: $SERVICE_NAME"
```

### 2. Build and Push Image

```bash
gcloud builds submit . --tag $AR_IMAGE_PATH --project $PROJECT_ID
```

### 3. Deploy to Cloud Run

```bash
gcloud run deploy $SERVICE_NAME \
  --image $AR_IMAGE_PATH \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 600 \
  --update-secrets="SUPABASE_URL=SUPABASE_URL:latest,SUPABASE_SERVICE_ROLE_KEY=SUPABASE_SERVICE_ROLE_KEY:latest" \
  --ingress=all \
  --project $PROJECT_ID
```

### 4. Get Service URL

```bash
gcloud run services describe $SERVICE_NAME --region $REGION --project $PROJECT_ID --format='value(status.url)'
```

---

## Deployment (Windows PowerShell)

```powershell
# --- Define variables ---
$PROJECT_ID = "creators-multi-verse"
$REGION = "europe-central2"
$AR_REPO = "creatorsm-docker-repo"
$IMAGE_NAME = "media-processing-service"
$IMAGE_TAG = $(Get-Date -Format yyyyMMdd-HHmmss)
$AR_IMAGE_PATH = "${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"
$SERVICE_NAME = "media-processing-svc"

# Build
gcloud builds submit . --tag $AR_IMAGE_PATH --project $PROJECT_ID

# Deploy
gcloud run deploy $SERVICE_NAME --image $AR_IMAGE_PATH --platform managed --region $REGION --allow-unauthenticated --port 8080 --memory 2Gi --cpu 2 --timeout 600 --update-secrets="SUPABASE_URL=SUPABASE_URL:latest,SUPABASE_SERVICE_ROLE_KEY=SUPABASE_SERVICE_ROLE_KEY:latest" --ingress=all --project $PROJECT_ID
```

---

## Quick Re-Deploy Script

For subsequent deployments, create `deploy.sh`:

```bash
#!/bin/bash
set -e

PROJECT_ID="creators-multi-verse"
REGION="europe-central2"
AR_REPO="creatorsm-docker-repo"
IMAGE_NAME="media-processing-service"
IMAGE_TAG=$(date +%Y%m%d-%H%M%S)
AR_IMAGE_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"
SERVICE_NAME="media-processing-svc"

echo "Building and deploying $IMAGE_NAME:$IMAGE_TAG..."

gcloud builds submit . --tag $AR_IMAGE_PATH --project $PROJECT_ID

gcloud run deploy $SERVICE_NAME \
  --image $AR_IMAGE_PATH \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 600 \
  --update-secrets="SUPABASE_URL=SUPABASE_URL:latest,SUPABASE_SERVICE_ROLE_KEY=SUPABASE_SERVICE_ROLE_KEY:latest" \
  --ingress=all \
  --project $PROJECT_ID

echo "Deployed successfully!"
gcloud run services describe $SERVICE_NAME --region $REGION --project $PROJECT_ID --format='value(status.url)'
```

---

## Testing

### Health Check
```bash
curl https://media-processing-svc-XXXXX-ew.a.run.app/health
```

### Export Video (example)
```bash
curl -X POST https://media-processing-svc-XXXXX-ew.a.run.app/video/export \
  -H "Content-Type: application/json" \
  -d '{
    "clips": [
      {
        "id": "clip1",
        "sourceUrl": "https://your-video-url.mp4",
        "sourceDuration": 10,
        "startTime": 0,
        "trimStart": 2,
        "trimEnd": 3
      }
    ],
    "userId": "user-uuid",
    "companyId": "company-uuid",
    "projectName": "Test Export"
  }'
```

---

## Environment Variables

| Variable | Description | Source |
|----------|-------------|--------|
| `SUPABASE_URL` | Supabase project URL | Secret Manager |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key | Secret Manager |
| `PORT` | Server port (default: 8080) | Cloud Run |

---

## Resources

- Memory: 2GB (for video processing)
- CPU: 2 cores
- Timeout: 600 seconds (10 minutes max for large exports)
- Concurrency: Default (80 requests per instance)

---

## Troubleshooting

### FFmpeg not found
Check the Docker build logs - FFmpeg should be installed via apt-get.

### Memory errors on large videos
Increase memory allocation: `--memory 4Gi`

### Timeout on long videos
Increase timeout: `--timeout 900` (15 minutes)

### Permission denied on Supabase
Verify the service has access to the secrets and the bucket policies allow uploads.
