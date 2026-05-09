#!/usr/bin/env bash
# deploy.sh - Build, push, and deploy to Google Kubernetes Engine
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# CONFIGURATION — Set these before running
# ──────────────────────────────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID env var}"
CLUSTER="heart-disease-cluster"
ZONE="us-central1-a"
IMAGE="gcr.io/${PROJECT_ID}/heart-disease-api"
TAG="${1:-latest}"

echo "🚀 Deploying Heart Disease API to GKE"
echo "   Project:  $PROJECT_ID"
echo "   Cluster:  $CLUSTER"
echo "   Zone:     $ZONE"
echo "   Image:    $IMAGE:$TAG"
echo ""

# ── Step 1: Enable APIs ──────────────────────────────────────
echo "📦 [1/8] Enabling required GCP APIs..."
gcloud services enable \
  container.googleapis.com \
  containerregistry.googleapis.com \
  --project "$PROJECT_ID" --quiet

# ── Step 2: Train model (if not present) ─────────────────────
if [ ! -f "models/best_model.joblib" ]; then
    echo "🧠 [2/8] Training model..."
    python -m src.train
else
    echo "✅ [2/8] Model already trained — skipping"
fi

# ── Step 3: Build Docker image ───────────────────────────────
echo "🐳 [3/8] Building Docker image..."
docker build -t "$IMAGE:$TAG" .

# ── Step 4: Authenticate & push to GCR ───────────────────────
echo "🔑 [4/8] Authenticating Docker with GCR..."
gcloud auth configure-docker --quiet
echo "📤 Pushing image to GCR..."
docker push "$IMAGE:$TAG"

# ── Step 5: Create GKE cluster (if not exists) ───────────────
echo "☸️  [5/8] Creating/connecting to GKE cluster..."
if gcloud container clusters describe "$CLUSTER" --zone "$ZONE" --project "$PROJECT_ID" &>/dev/null; then
    echo "   Cluster already exists — connecting..."
else
    echo "   Creating new cluster..."
    gcloud container clusters create "$CLUSTER" \
        --zone "$ZONE" \
        --project "$PROJECT_ID" \
        --num-nodes 2 \
        --machine-type e2-standard-2 \
        --enable-autoscaling \
        --min-nodes 1 \
        --max-nodes 4 \
        --quiet
fi

# ── Step 6: Get credentials ─────────────────────────────────
echo "🔗 [6/8] Getting cluster credentials..."
gcloud container clusters get-credentials "$CLUSTER" \
    --zone "$ZONE" \
    --project "$PROJECT_ID"

# ── Step 7: Apply Kubernetes manifests ───────────────────────
echo "📋 [7/8] Deploying to Kubernetes..."
sed "s|gcr.io/YOUR_GCR_PROJECT_ID/heart-disease-api:latest|${IMAGE}:${TAG}|g" \
    deployment/deployment.yaml | kubectl apply -f -
kubectl apply -f deployment/service.yaml
kubectl apply -f deployment/hpa.yaml

# ── Step 8: Wait & report ────────────────────────────────────
echo "⏳ [8/8] Waiting for rollout to complete..."
kubectl rollout status deployment/heart-disease-api --timeout=180s

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✅ DEPLOYMENT COMPLETE"
echo "═══════════════════════════════════════════════════════"

# Get external IP
for i in $(seq 1 20); do
    EXTERNAL_IP=$(kubectl get svc heart-disease-api-service \
        -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
    if [ -n "$EXTERNAL_IP" ]; then
        echo ""
        echo "  🌐 API URL:     http://$EXTERNAL_IP/predict"
        echo "  📖 Swagger UI:  http://$EXTERNAL_IP/docs"
        echo "  ❤️  Health:      http://$EXTERNAL_IP/health"
        echo "  📊 Metrics:     http://$EXTERNAL_IP/metrics"
        echo ""
        break
    fi
    echo "   Waiting for external IP... ($i/20)"
    sleep 10
done

if [ -z "$EXTERNAL_IP" ]; then
    echo "⚠️  External IP not yet assigned. Run:"
    echo "   kubectl get svc heart-disease-api-service"
fi
