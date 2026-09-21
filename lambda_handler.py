"""Lambda entry point (Function URL / API Gateway): POST {"question": "..."} -> {"answer": "..."}."""
import base64
import json

from rag_bedrock import rag_kb


def handler(event, context):
    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode()
    question = (json.loads(body).get("question") or "").strip()
    if not question:
        return {"statusCode": 400, "body": json.dumps({"error": "missing 'question'"})}
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"answer": rag_kb(question)}),
    }
