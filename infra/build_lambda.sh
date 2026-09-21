#!/usr/bin/env bash
# Bundle the Lambda package (code + Linux wheels) into infra/lambda_build.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf lambda_build && mkdir lambda_build
pip install --quiet --target lambda_build \
  --platform manylinux2014_x86_64 --python-version 3.13 --only-binary=:all: \
  boto3 "anthropic[bedrock]"
cp ../rag_bedrock.py ../lambda_handler.py lambda_build/
