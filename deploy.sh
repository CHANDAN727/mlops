#!/usr/bin/env bash
# deploy.sh - Build, push, and deploy to a GCP VM
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# CONFIGURATION — Set these before running
# ──────────────────────────────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID env var}"
VM_NAME="${GCP_VM_NAME:?Set GCP_VM_NAME env var}"
VM_ZONE="${GCP_VM_ZONE:-us-central1-a}"
IMAGE="gcr.io/${PROJECT_ID}/heart-disease-api"
TAG="${1:-latest}"

echo "🚀 Deploying Heart Disease API to GCP VM"
echo "   Project:  $PROJECT_ID"
echo "   VM:       $VM_NAME ($VM_ZONE)"
echo "   Image:    $IMAGE:$TAG"
echo ""

# ── Step 1: Enable APIs ──────────────────────────────────────
echo "📦 [1/6] Enabling required GCP APIs..."
gcloud services enable \
  compute.googleapis.com \
  containerregistry.googleapis.com \
  --project "$PROJECT_ID" --quiet

# ── Step 2: Train model (if not present) ─────────────────────
if [ ! -f "models/best_model.joblib" ]; then
    echo "🧠 [2/6] Training model..."
    python -m src.train
else
    echo "✅ [2/6] Model already trained — skipping"
fi

# ── Step 3: Build Docker image ───────────────────────────────
echo "🐳 [3/6] Building Docker image..."
docker build -t "$IMAGE:$TAG" .

# ── Step 4: Authenticate & push to GCR ───────────────────────
echo "🔑 [4/6] Authenticating Docker with GCR..."
gcloud auth configure-docker --quiet
echo "📤 Pushing image to GCR..."
docker push "$IMAGE:$TAG"

# ── Step 5: Deploy on VM ─────────────────────────────────────
echo "🖥️  [5/6] Deploying to VM..."
gcloud compute ssh "$VM_NAME" \
    --zone "$VM_ZONE" \
    --project "$PROJECT_ID" \
    --command="
        gcloud auth configure-docker --quiet
        docker pull $IMAGE:$TAG
        docker stop heart-disease-api 2>/dev/null || true
        docker rm heart-disease-api 2>/dev/null || true
        docker run -d \
            --name heart-disease-api \
            --restart unless-stopped \
            -p 8080:8080 \
            $IMAGE:$TAG
        sleep 5
        curl -sf http://localhost:8080/health && echo 'Container healthy!' || echo 'Health check pending...'
    "

# ── Step 6: Report ───────────────────────────────────────────
echo ""
EXTERNAL_IP=$(gcloud compute instances describe "$VM_NAME" \
    --zone "$VM_ZONE" \
    --project "$PROJECT_ID" \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo "════════════════════════════════════════════════════════"
echo "✅ Deployment Complete!"
echo "   API Docs:   http://$EXTERNAL_IP:8080/docs"
echo "   Health:     http://$EXTERNAL_IP:8080/health"
echo "   Predict:    http://$EXTERNAL_IP:8080/predict"
echo "   Metrics:    http://$EXTERNAL_IP:8080/metrics"
echo "════════════════════════════════════════════════════════"
