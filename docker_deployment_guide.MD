# --- Define your variables (FILLED WITH YOUR PROVIDED INFO) ---
# Run in PS #

# docker build -t creatorsm-agentic-pipeline .

$PROJECT_ID = "creators-multi-verse"
$REGION = "europe-central2"         # Assuming this is your preferred region based on previous examples
$AR_REPO = "my-docker-repo"         # You can reuse this from your 'langgraph-api'
$IMAGE_NAME = "gcs-openai-gemini-app" # <<< CHOOSE a descriptive name for this new image
$IMAGE_TAG = $(Get-Date -Format yyyyMMdd-HHmmss)
$AR_IMAGE_PATH = "${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

$SERVICE_NAME = "gcs-gemini-openai-app" # <<< SHORTER SERVICE NAME (20 chars)
$GCS_BUCKET_NAME_FOR_APP = "creatorsm-media-bucket" # Bucket your app uses

# Service Account for Cloud Build
$PROJECT_NUMBER = "1006785094868"
$CLOUD_BUILD_SA = "cr-gcs-gemini-openai-app-sa@creators-multi-verse.iam.gserviceaccount.com"

# Service Account for Cloud Run (we will create this)
$CLOUD_RUN_SA_NAME = "cr-${SERVICE_NAME}-sa" # e.g., cr-gcs-openai-gemini-sa
$CLOUD_RUN_SA_EMAIL = "${CLOUD_RUN_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Secret Names in Secret Manager (ASSUMING YOU NAMED THEM THIS WAY)
$GEMINI_SECRET_NAME = "GEMINI_API_KEY"
$OPENAI_SECRET_NAME = "OPENAI_API_KEY_VALUE"


# Port your application listens on (e.g., FastAPI/Uvicorn default)
$APP_PORT = 8000

# --- Phase 1: GCP Setup (Run these once per project/service setup, or verify) ---

# 1. Enable necessary APIs (if not already enabled)
Write-Host "Enabling APIs..."
gcloud services enable run.googleapis.com --project $PROJECT_ID
gcloud services enable cloudbuild.googleapis.com --project $PROJECT_ID
gcloud services enable artifactregistry.googleapis.com --project $PROJECT_ID
gcloud services enable iam.googleapis.com --project $PROJECT_ID
gcloud services enable storage.googleapis.com --project $PROJECT_ID
gcloud services enable secretmanager.googleapis.com --project $PROJECT_ID # For Secret Manager

# 2. Verify GCS Bucket exists (Assuming it does, as per your info)
Write-Host "Verifying GCS Bucket: $GCS_BUCKET_NAME_FOR_APP..."
If (-not (gcloud storage buckets list --project $PROJECT_ID | Where-Object { $_ -match "gs://${GCS_BUCKET_NAME_FOR_APP}/" })) {
    Write-Error "Bucket gs://$GCS_BUCKET_NAME_FOR_APP does not exist in project $PROJECT_ID! Please create it first."
    Exit 1
} Else {
    Write-Host "Bucket $GCS_BUCKET_NAME_FOR_APP found."
}

# 3. Create Artifact Registry Repo (if it doesn't exist)
Write-Host "Checking/Creating Artifact Registry Repo: $AR_REPO..."
If (-not (gcloud artifacts repositories describe $AR_REPO --project=$PROJECT_ID --location=$REGION --format="value(name)" 2>$null)) {
    Write-Host "Artifact Registry $AR_REPO does not exist. Creating..."
    gcloud artifacts repositories create $AR_REPO --repository-format=docker --location=$REGION --description="Docker repository for applications" --project $PROJECT_ID
    Write-Host "Artifact Registry $AR_REPO created."
} Else {
    Write-Host "Artifact Registry $AR_REPO already exists."
}

# 4. Create a dedicated Service Account for the Cloud Run service
Write-Host "Checking/Creating Cloud Run Service Account: $CLOUD_RUN_SA_EMAIL..."
If (-not (gcloud iam service-accounts list --project=$PROJECT_ID --filter="email:$CLOUD_RUN_SA_EMAIL" --format="value(email)")) {
    Write-Host "Service Account $CLOUD_RUN_SA_NAME does not exist. Creating..."
    gcloud iam service-accounts create $CLOUD_RUN_SA_NAME --display-name="Cloud Run SA for $SERVICE_NAME" --project $PROJECT_ID
    Write-Host "Service Account $CLOUD_RUN_SA_NAME created."
} Else {
    Write-Host "Service Account $CLOUD_RUN_SA_NAME already exists."
}

# 5. Grant the Cloud Run SA permissions:
# 5a. On the GCS Bucket
Write-Host "Granting GCS permissions to $CLOUD_RUN_SA_EMAIL for bucket $GCS_BUCKET_NAME_FOR_APP..."
gcloud storage buckets add-iam-policy-binding gs://$GCS_BUCKET_NAME_FOR_APP --member="serviceAccount:$CLOUD_RUN_SA_EMAIL" --role="roles/storage.objectAdmin" --project $PROJECT_ID --condition=None

# 5b. To access secrets in Secret Manager
Write-Host "Granting Secret Accessor role to $CLOUD_RUN_SA_EMAIL for project $PROJECT_ID..."
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$CLOUD_RUN_SA_EMAIL" --role="roles/secretmanager.secretAccessor" --condition=None
# More granular:
# gcloud secrets add-iam-policy-binding $GEMINI_SECRET_NAME --project=$PROJECT_ID --member="serviceAccount:$CLOUD_RUN_SA_EMAIL" --role="roles/secretmanager.secretAccessor"
# gcloud secrets add-iam-policy-binding $OPENAI_SECRET_NAME --project=$PROJECT_ID --member="serviceAccount:$CLOUD_RUN_SA_EMAIL" --role="roles/secretmanager.secretAccessor"


# 6. Grant Permissions to Cloud Build SA ($CLOUD_BUILD_SA)

# 6a. Allow Cloud Build SA to push to Artifact Registry
Write-Host "Granting Artifact Registry Writer role to Cloud Build SA: $CLOUD_BUILD_SA..."
gcloud artifacts repositories add-iam-policy-binding $AR_REPO --project=$PROJECT_ID --location=$REGION --member="serviceAccount:$CLOUD_BUILD_SA" --role="roles/artifactregistry.writer" --condition=None

# 6b. Allow Cloud Build SA to deploy to Cloud Run and manage the service
Write-Host "Granting Cloud Run Admin role to Cloud Build SA: $CLOUD_BUILD_SA..."
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$CLOUD_BUILD_SA" --role="roles/run.admin" --condition=None

# 6c. Allow Cloud Build SA to act as (impersonate) the Cloud Run Service Account during deployment
Write-Host "Granting IAM Service Account User role to Cloud Build SA ($CLOUD_BUILD_SA) on Cloud Run SA ($CLOUD_RUN_SA_EMAIL)..."
gcloud iam service-accounts add-iam-policy-binding $CLOUD_RUN_SA_EMAIL --project=$PROJECT_ID --member="serviceAccount:$CLOUD_BUILD_SA" --role="roles/iam.serviceAccountUser" --condition=None

# 6d. Grant Logs Writer Permission to Cloud Build SA (if not already done broadly)
Write-Host "Granting Logs Writer role to Cloud Build SA: $CLOUD_BUILD_SA..."
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$CLOUD_BUILD_SA" --role="roles/logging.logWriter" --condition=None

Write-Host "--- One-time setup steps complete (or verified) ---"
# Pause # Optional: pause before build and deploy

# --- Phase 3: Build, Push, and Deploy Flow ---

# 7. Submit the build to Cloud Build (Builds & Pushes to Artifact Registry)
Write-Host "Submitting build to Cloud Build for image $AR_IMAGE_PATH..."
# Make sure your Dockerfile is in the current directory (.)
gcloud builds submit . --tag $AR_IMAGE_PATH --project $PROJECT_ID



############################################################################

# 8. Deploy the new image to Google Cloud Run
Write-Host "Deploying to Cloud Run service: $SERVICE_NAME..."
# One Line
gcloud run deploy $SERVICE_NAME --image $AR_IMAGE_PATH --platform managed --region $REGION --allow-unauthenticated --port $APP_PORT --service-account $CLOUD_RUN_SA_EMAIL --set-env-vars="GCS_BUCKET_NAME=$GCS_BUCKET_NAME_FOR_APP,GCS_PROJECT_ID=$PROJECT_ID" --update-secrets="GEMINI_API_KEY=${GEMINI_SECRET_NAME}:latest,OPENAI_API_KEY_VALUE=${OPENAI_SECRET_NAME}:latest" --project $PROJECT_ID


Write-Host "Deployment commands executed for $SERVICE_NAME."
Write-Host "Service URL will be available in the Cloud Run console for project $PROJECT_ID in region $REGION."

# With VPC Connector
gcloud run deploy $SERVICE_NAME --image $AR_IMAGE_PATH --platform managed --region $REGION --allow-unauthenticated --port $APP_PORT --service-account $CLOUD_RUN_SA_EMAIL --set-env-vars="GCS_BUCKET_NAME=$GCS_BUCKET_NAME_FOR_APP,GCS_PROJECT_ID=$PROJECT_ID" --update-secrets="GEMINI_API_KEY=${GEMINI_SECRET_NAME}:latest,OPENAI_API_KEY_VALUE=${OPENAI_SECRET_NAME}:latest" --vpc-connector vpc-con-openai --vpc-egress all --project $PROJECT_ID
