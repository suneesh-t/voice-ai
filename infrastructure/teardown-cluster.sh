#!/usr/bin/env bash
set -euo pipefail

#
# Tear down the entire voice-ai EKS infrastructure.
#
# Deletes in reverse order:
#   1. RayService deployments
#   2. KubeRay operator
#   3. Karpenter (NodePools, EC2NodeClass, Helm release, CloudFormation stack)
#   4. NVIDIA device plugin
#   5. CloudWatch add-on
#   6. ECR repositories
#   7. EKS cluster
#
# Usage:
#   cd infrastructure
#   bash teardown-cluster.sh
#

CLUSTER_NAME="model-management"
REGION="us-east-1"

export AWS_PAGER=""

echo "============================================"
echo "  Voice AI — Teardown"
echo "============================================"
echo ""
echo "WARNING: This will DELETE the entire ${CLUSTER_NAME} cluster"
echo "         and all resources in ${REGION}."
echo ""
read -rp "Type 'yes' to confirm: " CONFIRM
if [ "${CONFIRM}" != "yes" ]; then
  echo "Aborted."
  exit 0
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# ---------------------------------------------------------------
# Step 1: Delete RayService deployments
# ---------------------------------------------------------------
echo ""
echo "=== Step 1/7: Deleting RayService deployments ==="
kubectl delete rayservice --all -n default 2>/dev/null || true
kubectl delete serviceaccount cohere-transcribe-sa -n default 2>/dev/null || true
kubectl delete serviceaccount qwen3-tts-sa -n default 2>/dev/null || true
kubectl delete secret hf-token -n default 2>/dev/null || true

echo "Waiting for Ray pods to terminate..."
kubectl wait --for=delete pod -l ray.io/cluster --timeout=120s -n default 2>/dev/null || true

# ---------------------------------------------------------------
# Step 2: Delete KubeRay operator
# ---------------------------------------------------------------
echo ""
echo "=== Step 2/7: Uninstalling KubeRay operator ==="
helm uninstall kuberay-operator -n kuberay-system 2>/dev/null || true
kubectl delete namespace kuberay-system 2>/dev/null || true

# ---------------------------------------------------------------
# Step 3: Delete Karpenter resources
# ---------------------------------------------------------------
echo ""
echo "=== Step 3/7: Removing Karpenter ==="
kubectl delete nodepool --all 2>/dev/null || true
kubectl delete ec2nodeclass --all 2>/dev/null || true

echo "Waiting for Karpenter nodes to drain..."
sleep 30

helm uninstall karpenter -n kube-system 2>/dev/null || true

eksctl delete iamserviceaccount \
  --cluster "${CLUSTER_NAME}" \
  --region "${REGION}" \
  --name karpenter \
  --namespace kube-system 2>/dev/null || true

# Delete IAM instance profile and role
ROLE_NAME="KarpenterNodeRole-${CLUSTER_NAME}"
aws iam remove-role-from-instance-profile \
  --instance-profile-name "${ROLE_NAME}" \
  --role-name "${ROLE_NAME}" 2>/dev/null || true
aws iam delete-instance-profile \
  --instance-profile-name "${ROLE_NAME}" 2>/dev/null || true

for POLICY_ARN in \
  "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy" \
  "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy" \
  "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly" \
  "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"; do
  aws iam detach-role-policy --role-name "${ROLE_NAME}" \
    --policy-arn "${POLICY_ARN}" 2>/dev/null || true
done
aws iam delete-role --role-name "${ROLE_NAME}" 2>/dev/null || true

aws cloudformation delete-stack \
  --stack-name "Karpenter-${CLUSTER_NAME}" \
  --region "${REGION}" 2>/dev/null || true

# ---------------------------------------------------------------
# Step 4: Delete NVIDIA device plugin
# ---------------------------------------------------------------
echo ""
echo "=== Step 4/7: Removing NVIDIA device plugin ==="
helm uninstall nvidia-device-plugin -n kube-system 2>/dev/null || true

# ---------------------------------------------------------------
# Step 5: Delete CloudWatch add-on
# ---------------------------------------------------------------
echo ""
echo "=== Step 5/7: Removing CloudWatch Container Insights ==="
aws eks delete-addon \
  --cluster-name "${CLUSTER_NAME}" \
  --region "${REGION}" \
  --addon-name amazon-cloudwatch-observability 2>/dev/null || true

eksctl delete iamserviceaccount \
  --cluster "${CLUSTER_NAME}" \
  --region "${REGION}" \
  --name cloudwatch-agent \
  --namespace amazon-cloudwatch 2>/dev/null || true

aws iam delete-role \
  --role-name "CloudWatchAgentRole-${CLUSTER_NAME}" 2>/dev/null || true

# ---------------------------------------------------------------
# Step 6: Delete ECR repositories
# ---------------------------------------------------------------
echo ""
echo "=== Step 6/7: Deleting ECR repositories ==="
for REPO in cohere-transcribe-serve qwen3-tts-serve; do
  aws ecr delete-repository \
    --repository-name "${REPO}" \
    --region "${REGION}" \
    --force 2>/dev/null || true
  echo "  Deleted ECR repo: ${REPO}"
done

# ---------------------------------------------------------------
# Step 7: Delete EKS cluster
# ---------------------------------------------------------------
echo ""
echo "=== Step 7/7: Deleting EKS cluster (takes ~10 minutes) ==="
eksctl delete cluster --name "${CLUSTER_NAME}" --region "${REGION}"

echo ""
echo "============================================"
echo "  Teardown complete"
echo "============================================"
