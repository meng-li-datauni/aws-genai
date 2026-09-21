"""Lambda entry point (HTTP API / Function URL).

POST {"question": "...", "history": [{"role": "user"|"assistant", "content": "..."}]}
  -> {"answer": "..."}
"""
import base64
import json

from rag_bedrock import rag_kb

MAX_QUESTION_CHARS = 1000


def _response(status: int, payload: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def handler(event, context):
    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return _response(400, {"error": "invalid JSON"})
    question = (data.get("question") or "").strip() if isinstance(data, dict) else ""
    if not question:
        return _response(400, {"error": "missing 'question'"})
    if len(question) > MAX_QUESTION_CHARS:
        return _response(400, {"error": f"question longer than {MAX_QUESTION_CHARS} characters"})
    return _response(200, {"answer": rag_kb(question, data.get("history"))})
