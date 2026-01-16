#!/bin/bash
# Media Processing Service - Quick Deploy Script
set -e

PROJECT_ID="creators-multi-verse"
REGION="europe-central2"
AR_REPO="creatorsm-docker-repo"
IMAGE_NAME="media-processing-service"
IMAGE_TAG=$(date +%Y%m%d-%H%M%S)
AR_IMAGE_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"
SERVICE_NAME="media-processing-svc"

echo "========================================"
echo "Media Processing Service Deployment"
echo "========================================"
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"
echo "========================================"

echo ""
echo "Step 1: Building and pushing image..."
gcloud builds submit . --tag $AR_IMAGE_PATH --project $PROJECT_ID

echo ""
echo "Step 2: Deploying to Cloud Run..."
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

echo ""
echo "========================================"
echo "Deployment Complete!"
echo "========================================"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --project $PROJECT_ID --format='value(status.url)')
echo "Service URL: $SERVICE_URL"
echo ""
echo "Test with: curl $SERVICE_URL/health"
