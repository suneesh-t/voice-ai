#!/usr/bin/env bash
set -euo pipefail

#
# Full EKS cluster setup for the voice-ai platform.
#
# Creates the model-management EKS cluster with all required components:
#   1. EKS cluster (eksctl)
#   2. NVIDIA device plugin (GPU scheduling)
#   3. Karpenter (GPU node auto-provisioning)
#   4. KubeRay operator (Ray Serve lifecycle management)
#   5. CloudWatch Container Insights (observability)
#
# Prerequisites:
#   - AWS CLI v2 configured with admin-level IAM permissions
#   - eksctl >= 0.200.0
#   - kubectl >= 1.32
#   - helm >= 3.16
#   - docker running (for service deployments later)
#
# Usage:
#   cd infrastructure
#   bash setup-cluster.sh
#

CLUSTER_NAME="model-management"
REGION="us-east-1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export AWS_PAGER=""

echo "============================================"
echo "  Voice AI — EKS Cluster Setup"
echo "============================================"
echo "Cluster: ${CLUSTER_NAME}"
echo "Region:  ${REGION}"
echo ""

# ---------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------
echo "=== Pre-flight checks ==="
for cmd in aws eksctl kubectl helm docker; do
  if ! command -v "${cmd}" &>/dev/null; then
    echo "ERROR: ${cmd} is not installed or not in PATH"
    exit 1
  fi
  echo "  ✓ ${cmd}"
done

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "  ✓ AWS Account: ${AWS_ACCOUNT_ID}"
echo ""

# ---------------------------------------------------------------
# Step 1: Create EKS cluster
# ---------------------------------------------------------------
echo "=== Step 1/5: Creating EKS cluster ==="

EXISTING_CLUSTER=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${REGION}" \
  --query "cluster.status" --output text 2>/dev/null || echo "NOT_FOUND")

if [ "${EXISTING_CLUSTER}" = "ACTIVE" ]; then
  echo "Cluster ${CLUSTER_NAME} already exists and is ACTIVE. Skipping creation."
  aws eks update-kubeconfig --name "${CLUSTER_NAME}" --region "${REGION}"
else
  echo "Creating cluster ${CLUSTER_NAME} (this takes ~15-20 minutes)..."
  eksctl create cluster -f "${SCRIPT_DIR}/eks/cluster-config.yaml"
fi
echo ""

# ---------------------------------------------------------------
# Step 2: Install NVIDIA device plugin
# ---------------------------------------------------------------
echo "=== Step 2/5: Installing NVIDIA device plugin ==="
bash "${SCRIPT_DIR}/nvidia/install-nvidia-plugin.sh"
echo ""

# ---------------------------------------------------------------
# Step 3: Install Karpenter + NodeClass + GPU NodePools
# ---------------------------------------------------------------
echo "=== Step 3/5: Installing Karpenter ==="
bash "${SCRIPT_DIR}/karpenter/install-karpenter.sh"

echo "Applying EC2NodeClass..."
kubectl apply -f "${SCRIPT_DIR}/karpenter/ec2-nodeclass.yaml"

echo "Applying GPU NodePools..."
kubectl apply -f "${SCRIPT_DIR}/../cohere-transcribe/karpenter-gpu-nodepool.yaml"
kubectl apply -f "${SCRIPT_DIR}/../qwen3-tts/karpenter-gpu-nodepool.yaml"
echo ""

# ---------------------------------------------------------------
# Step 4: Install KubeRay operator
# ---------------------------------------------------------------
echo "=== Step 4/5: Installing KubeRay operator ==="
bash "${SCRIPT_DIR}/kuberay/install-kuberay.sh"
echo ""

# ---------------------------------------------------------------
# Step 5: Enable Container Insights (optional but recommended)
# ---------------------------------------------------------------
echo "=== Step 5/5: Enabling CloudWatch Container Insights ==="
bash "${SCRIPT_DIR}/monitoring/container-insights.sh"
echo ""

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo "============================================"
echo "  Setup Complete"
echo "============================================"
echo ""
echo "Cluster:    ${CLUSTER_NAME}"
echo "Region:     ${REGION}"
echo "Account:    ${AWS_ACCOUNT_ID}"
echo ""
echo "Components installed:"
echo "  ✓ EKS cluster (v1.32)"
echo "  ✓ NVIDIA device plugin"
echo "  ✓ Karpenter (v1.3.0) + EC2NodeClass + GPU NodePools"
echo "  ✓ KubeRay operator (v1.3.0)"
echo "  ✓ CloudWatch Container Insights"
echo ""
echo "Next steps — deploy services:"
echo "  export HF_TOKEN=hf_xxxxx"
echo "  cd cohere-transcribe && bash deploy.sh"
echo "  cd qwen3-tts && bash deploy.sh"
echo ""
echo "Verify cluster:"
echo "  kubectl get nodes"
echo "  kubectl get pods -n kube-system"
echo "  kubectl get pods -n kuberay-system"
