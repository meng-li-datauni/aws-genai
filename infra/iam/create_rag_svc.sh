#!/usr/bin/env bash
# Create the rag-svc IAM user. Run with ADMIN credentials (not website-svc).
set -euo pipefail
export REGION="${AWS_REGION:-eu-west-3}"
export ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
cd "$(dirname "$0")"

envsubst < rag-svc-policy.json > /tmp/rag-svc-policy.rendered.json
aws iam create-user --user-name rag-svc
aws iam put-user-policy --user-name rag-svc --policy-name rag-svc \
  --policy-document file:///tmp/rag-svc-policy.rendered.json
rm /tmp/rag-svc-policy.rendered.json

# The secret is shown once - store it straight into a named profile, not in the repo.
aws iam create-access-key --user-name rag-svc --query 'AccessKey.[AccessKeyId,SecretAccessKey]' --output text
echo "Next: aws configure --profile rag-svc   (region: $REGION)"
echo "Then, still as admin, once: AWS_PROFILE=<admin> cdk bootstrap aws://$ACCOUNT_ID/$REGION"
