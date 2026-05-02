#!/usr/bin/env bash
set -euo pipefail

#
# Deploy Whisper Large-v3-Turbo STT to EKS (model-management cluster).
#
# Steps:
#   1. Create ECR repository
#   2. Build & push Docker image
#   3. Update kubeconfig
#   4. Apply Karpenter GPU NodePool
#   5. Create HuggingFace token secret
#   6. Deploy RayService
#

REGION="us-east-1"
CLUSTER="model-management"
ECR_REPO="whisper-v3-turbo-serve"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export AWS_PAGER=""

IMAGE_TAG="$(git rev-parse --short HEAD)-$(date +%Y%m%d%H%M%S)"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

FULL_IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

echo "============================================"
echo "  Whisper Large-v3-Turbo — Deploy to EKS"
echo "============================================"
echo "Account:  ${AWS_ACCOUNT_ID}"
echo "Region:   ${REGION}"
echo "Cluster:  ${CLUSTER}"
echo "Image:    ${FULL_IMAGE}"
echo ""

# ---------------------------------------------------------------
# Step 1: ECR repository
# ---------------------------------------------------------------
echo "=== [1/6] Creating ECR repository ==="
aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${REGION}" 2>/dev/null || \
  aws ecr create-repository --repository-name "${ECR_REPO}" --region "${REGION}" \
    --image-scanning-configuration scanOnPush=true

# ---------------------------------------------------------------
# Step 2: Build & push Docker image
# ---------------------------------------------------------------
echo "=== [2/6] Building and pushing Docker image ==="
aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
docker build --platform linux/amd64 -t "${FULL_IMAGE}" "${SCRIPT_DIR}"
docker push "${FULL_IMAGE}"

# ---------------------------------------------------------------
# Step 3: Update kubeconfig
# ---------------------------------------------------------------
echo "=== [3/6] Updating kubeconfig ==="
aws eks update-kubeconfig --name "${CLUSTER}" --region "${REGION}"

# ---------------------------------------------------------------
# Step 4: Karpenter GPU NodePool
# ---------------------------------------------------------------
echo "=== [4/6] Applying Karpenter GPU NodePool ==="
kubectl apply -f "${SCRIPT_DIR}/karpenter-gpu-nodepool.yaml"

# ---------------------------------------------------------------
# Step 5: HuggingFace token secret
# ---------------------------------------------------------------
echo "=== [5/6] Creating HuggingFace token secret ==="
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: Set HF_TOKEN environment variable with your HuggingFace access token"
  echo "  export HF_TOKEN=hf_xxxxx"
  exit 1
fi
kubectl create secret generic hf-token \
  --from-file=hf_token=/dev/stdin \
  --dry-run=client -o yaml <<< "${HF_TOKEN}" | kubectl apply -f -

# ---------------------------------------------------------------
# Step 6: Deploy RayService
# ---------------------------------------------------------------
echo "=== [6/6] Deploying RayService ==="
sed -e "s|<AWS_ACCOUNT_ID>|${AWS_ACCOUNT_ID}|g" \
    -e "s|<IMAGE_TAG>|${IMAGE_TAG}|g" \
    "${SCRIPT_DIR}/ray-service-whisper.yaml" | kubectl apply -f -

echo ""
echo "============================================"
echo "  Deployment initiated! (image tag: ${IMAGE_TAG})"
echo "============================================"
echo ""
echo "Monitor:"
echo "  kubectl get rayservice whisper-v3-turbo-service"
echo "  kubectl get pods -l ray.io/cluster"
echo "  kubectl logs -l ray.io/node-type=worker -c ray-worker --tail=50 -f"
echo ""
echo "Test (after port-forward):"
echo "  kubectl port-forward svc/whisper-v3-turbo-service-serve-svc 8003:8000"
echo "  curl -X POST http://localhost:8003/ -F 'file=@test_audio.wav'"
