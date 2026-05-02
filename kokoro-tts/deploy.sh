#!/usr/bin/env bash
set -euo pipefail

export AWS_PAGER=""

# ============================================================
# Deploy hexgrad/Kokoro-82M on EKS
# 82M-param StyleTTS2 TTS model, in-process PyTorch via Ray Serve.
# Designed to share a single GPU across multiple replicas (num_gpus=0.5).
#
# Prerequisites:
#   - aws CLI configured
#   - kubectl configured for model-management cluster
#   - docker running
#   - HF_TOKEN env var (Kokoro is Apache-2.0 / public, but misaki + HF cache
#     are friendlier with a token; reuses shared hf-token secret)
# ============================================================

REGION="us-east-1"
CLUSTER_NAME="model-management"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="kokoro-tts-serve"
IMAGE_TAG="$(git rev-parse --short HEAD 2>/dev/null || echo "build")-$(date +%Y%m%d%H%M%S)"
FULL_IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Deploying kokoro-tts (Kokoro-82M) to EKS"
echo "Account:  ${AWS_ACCOUNT_ID}"
echo "Region:   ${REGION}"
echo "Cluster:  ${CLUSTER_NAME}"
echo "Image:    ${FULL_IMAGE}"
echo "=========================================="

# --- Step 1: Create ECR repository ---
echo -e "\n[1/6] Creating ECR repository..."
aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${REGION}" 2>/dev/null || \
  aws ecr create-repository --repository-name "${ECR_REPO}" --region "${REGION}" --image-scanning-configuration scanOnPush=true

# --- Step 2: Build and push Docker image ---
echo -e "\n[2/6] Building and pushing Docker image..."
aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

docker build --platform linux/amd64 -t "${ECR_REPO}:${IMAGE_TAG}" "${SCRIPT_DIR}"
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${FULL_IMAGE}"
docker push "${FULL_IMAGE}"

# --- Step 3: Update kubeconfig ---
echo -e "\n[3/6] Updating kubeconfig..."
aws eks update-kubeconfig --name "${CLUSTER_NAME}" --region "${REGION}"

# --- Step 4: Apply GPU NodePool ---
echo -e "\n[4/6] Creating Karpenter GPU NodePool..."
kubectl apply -f "${SCRIPT_DIR}/karpenter-gpu-nodepool.yaml"

# --- Step 5: Create HF token secret (shared across services) ---
echo -e "\n[5/6] Ensuring HuggingFace token secret..."
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: Set HF_TOKEN environment variable with your HuggingFace access token"
  echo "  export HF_TOKEN=hf_xxxxx"
  exit 1
fi

echo -n "${HF_TOKEN}" | kubectl create secret generic hf-token \
  --from-file=hf_token=/dev/stdin \
  --namespace default \
  --dry-run=client -o yaml | kubectl apply -f -

# --- Step 6: Deploy RayService ---
echo -e "\n[6/6] Deploying RayService..."
sed -e "s|<AWS_ACCOUNT_ID>|${AWS_ACCOUNT_ID}|g" \
    -e "s|<IMAGE_TAG>|${IMAGE_TAG}|g" \
    "${SCRIPT_DIR}/ray-service-kokoro.yaml" | kubectl apply -f -

echo -e "\n=========================================="
echo "Deployment initiated! (image tag: ${IMAGE_TAG})"
echo ""
echo "Monitor progress:"
echo "  kubectl get rayservice kokoro-tts -w"
echo "  kubectl get pods -l ray.io/cluster -w"
echo ""
echo "Once running, port-forward to test:"
echo "  kubectl port-forward svc/kokoro-tts-serve-svc 8004:8000"
echo ""
echo "Test (streaming PCM):"
echo '  curl -N -X POST http://localhost:8004/v1/audio/speech \'
echo '    -H "Content-Type: application/json" \'
echo '    -d "{\"input\":\"Hello from Kokoro.\",\"voice\":\"af_heart\",\"stream\":true,\"response_format\":\"pcm\"}" \'
echo '    --output test.pcm'
echo ""
echo "Test (WAV file):"
echo '  curl -X POST http://localhost:8004/v1/audio/speech \'
echo '    -H "Content-Type: application/json" \'
echo '    -d "{\"input\":\"Hello from Kokoro.\",\"voice\":\"af_heart\"}" \'
echo '    --output test.wav'
echo "=========================================="
