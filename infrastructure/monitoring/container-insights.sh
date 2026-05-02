#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="model-management"
REGION="us-east-1"

export AWS_PAGER=""

echo "=== Enabling CloudWatch Container Insights on ${CLUSTER_NAME} ==="

# ---------------------------------------------------------------
# Step 1: Create IRSA for CloudWatch Agent
# ---------------------------------------------------------------
echo "Creating IRSA for CloudWatch Agent..."
eksctl create iamserviceaccount \
  --cluster "${CLUSTER_NAME}" \
  --region "${REGION}" \
  --name cloudwatch-agent \
  --namespace amazon-cloudwatch \
  --role-name "CloudWatchAgentRole-${CLUSTER_NAME}" \
  --attach-policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy \
  --override-existing-serviceaccounts \
  --approve

# ---------------------------------------------------------------
# Step 2: Install CloudWatch Observability add-on
# ---------------------------------------------------------------
echo "Installing CloudWatch Observability EKS add-on..."
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

aws eks create-addon \
  --cluster-name "${CLUSTER_NAME}" \
  --region "${REGION}" \
  --addon-name amazon-cloudwatch-observability \
  --service-account-role-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:role/CloudWatchAgentRole-${CLUSTER_NAME}" \
  2>/dev/null || \
aws eks update-addon \
  --cluster-name "${CLUSTER_NAME}" \
  --region "${REGION}" \
  --addon-name amazon-cloudwatch-observability \
  --service-account-role-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:role/CloudWatchAgentRole-${CLUSTER_NAME}"

echo ""
echo "=== Container Insights enabled ==="
echo "Metrics available in CloudWatch under:"
echo "  Namespace: ContainerInsights"
echo "  Cluster:   ${CLUSTER_NAME}"
echo ""
echo "Logs available in CloudWatch Logs:"
echo "  /aws/containerinsights/${CLUSTER_NAME}/application"
echo "  /aws/containerinsights/${CLUSTER_NAME}/performance"
