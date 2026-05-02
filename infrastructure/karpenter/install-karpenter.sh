#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="model-management"
REGION="us-east-1"
KARPENTER_VERSION="1.3.0"

export AWS_PAGER=""

echo "=== Installing Karpenter v${KARPENTER_VERSION} on ${CLUSTER_NAME} ==="

# ---------------------------------------------------------------
# Step 1: Get cluster details
# ---------------------------------------------------------------
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
CLUSTER_ENDPOINT=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${REGION}" \
  --query "cluster.endpoint" --output text)

echo "Account:  ${AWS_ACCOUNT_ID}"
echo "Cluster:  ${CLUSTER_NAME}"
echo "Endpoint: ${CLUSTER_ENDPOINT}"

# ---------------------------------------------------------------
# Step 2: Create Karpenter IAM roles via CloudFormation
# ---------------------------------------------------------------
TEMPOUT=$(mktemp)
curl -fsSL "https://raw.githubusercontent.com/aws/karpenter-provider-aws/v${KARPENTER_VERSION}/website/content/en/docs/getting-started/getting-started-with-karpenter/cloudformation.yaml" > "${TEMPOUT}"

echo "Creating Karpenter CloudFormation stack..."
aws cloudformation deploy \
  --stack-name "Karpenter-${CLUSTER_NAME}" \
  --template-file "${TEMPOUT}" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "${REGION}" \
  --parameter-overrides \
    "ClusterName=${CLUSTER_NAME}" || true

# ---------------------------------------------------------------
# Step 3: Tag subnets for Karpenter discovery
# ---------------------------------------------------------------
echo "Tagging subnets for Karpenter discovery..."
SUBNET_IDS=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${REGION}" \
  --query "cluster.resourcesVpcConfig.subnetIds" --output text)

for SUBNET_ID in ${SUBNET_IDS}; do
  aws ec2 create-tags --resources "${SUBNET_ID}" --region "${REGION}" \
    --tags "Key=karpenter.sh/discovery,Value=${CLUSTER_NAME}" 2>/dev/null || true
done

# Tag security groups
CLUSTER_SG=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${REGION}" \
  --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text)
aws ec2 create-tags --resources "${CLUSTER_SG}" --region "${REGION}" \
  --tags "Key=karpenter.sh/discovery,Value=${CLUSTER_NAME}" 2>/dev/null || true

# ---------------------------------------------------------------
# Step 4: Create Karpenter IRSA (IAM Role for Service Account)
# ---------------------------------------------------------------
echo "Creating Karpenter IRSA..."
eksctl create iamserviceaccount \
  --cluster "${CLUSTER_NAME}" \
  --region "${REGION}" \
  --name karpenter \
  --namespace kube-system \
  --role-name "KarpenterControllerRole-${CLUSTER_NAME}" \
  --attach-policy-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:policy/KarpenterControllerPolicy-${CLUSTER_NAME}" \
  --override-existing-serviceaccounts \
  --approve

# ---------------------------------------------------------------
# Step 5: Create EC2NodeClass IAM instance profile
# ---------------------------------------------------------------
echo "Creating Karpenter EC2NodeClass instance profile..."
ROLE_NAME="KarpenterNodeRole-${CLUSTER_NAME}"

aws iam create-role \
  --role-name "${ROLE_NAME}" \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }' 2>/dev/null || true

aws iam attach-role-policy --role-name "${ROLE_NAME}" \
  --policy-arn arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy 2>/dev/null || true
aws iam attach-role-policy --role-name "${ROLE_NAME}" \
  --policy-arn arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy 2>/dev/null || true
aws iam attach-role-policy --role-name "${ROLE_NAME}" \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly 2>/dev/null || true
aws iam attach-role-policy --role-name "${ROLE_NAME}" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore 2>/dev/null || true

aws iam create-instance-profile --instance-profile-name "${ROLE_NAME}" 2>/dev/null || true
aws iam add-role-to-instance-profile --instance-profile-name "${ROLE_NAME}" --role-name "${ROLE_NAME}" 2>/dev/null || true

# Map the node role into aws-auth ConfigMap
eksctl create iamidentitymapping \
  --cluster "${CLUSTER_NAME}" \
  --region "${REGION}" \
  --arn "arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}" \
  --group system:bootstrappers \
  --group system:nodes \
  --username system:node:{{EC2PrivateDNSName}} 2>/dev/null || true

# ---------------------------------------------------------------
# Step 6: Install Karpenter via Helm
# ---------------------------------------------------------------
echo "Installing Karpenter via Helm..."
helm upgrade --install karpenter oci://public.ecr.aws/karpenter/karpenter \
  --version "${KARPENTER_VERSION}" \
  --namespace kube-system \
  --set "settings.clusterName=${CLUSTER_NAME}" \
  --set "settings.interruptionQueue=Karpenter-${CLUSTER_NAME}" \
  --set "settings.clusterEndpoint=${CLUSTER_ENDPOINT}" \
  --set "serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn=arn:aws:iam::${AWS_ACCOUNT_ID}:role/KarpenterControllerRole-${CLUSTER_NAME}" \
  --wait

echo ""
echo "=== Karpenter installed ==="
echo "Next: apply EC2NodeClass and NodePools"
echo "  kubectl apply -f ../karpenter/ec2-nodeclass.yaml"
echo "  kubectl apply -f ../../cohere-transcribe/karpenter-gpu-nodepool.yaml"
echo "  kubectl apply -f ../../qwen3-tts/karpenter-gpu-nodepool.yaml"
