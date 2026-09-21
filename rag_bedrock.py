"""RAG on AWS with Amazon Bedrock, using ai-professional-01.pdf as the source document.

Two ways to run it:

  local   No AWS infrastructure to set up. Reads the PDF, chunks it, embeds the chunks with
          Titan Embeddings and searches them in memory. Good for trying things out.
  kb      Managed. Uploads the PDF to S3 and queries a Bedrock Knowledge Base
          (chunking, embedding and vector search are handled by AWS). Use this for deployment.

Environment:
  AWS_REGION          e.g. us-east-1
  PDF_PATH            default: ~/Downloads/ai-professional-01.pdf
  S3_BUCKET           kb only: bucket the Knowledge Base data source points at
  KNOWLEDGE_BASE_ID   kb only
  DATA_SOURCE_ID      kb only, for `ingest`
  MAX_OUTPUT_TOKENS   cap on the answer length in tokens (default 1500)

Usage:
  python rag_bedrock.py local "What domains does the exam cover?"
  python rag_bedrock.py upload            # copy the PDF to S3
  python rag_bedrock.py ingest            # sync S3 -> Knowledge Base
  python rag_bedrock.py kb    "What domains does the exam cover?"
"""
from __future__ import annotations

import json
import os
import sys

import boto3
from anthropic import AnthropicBedrockMantle

REGION = os.environ.get("AWS_REGION", "us-east-1")
PDF_PATH = os.path.expanduser(os.environ.get("PDF_PATH", "~/Downloads/ai-professional-01.pdf"))
KB_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "1500"))  # hard cap; includes thinking tokens
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
GENERATION_MODEL = "anthropic.claude-opus-5"  # Bedrock ID = "anthropic." + first-party ID

Chunk = dict  # {"source": str, "text": str}


# --------------------------------------------------------------------------- generation
def generate(question: str, chunks: list[Chunk]) -> str:
    """Answer the question from the retrieved chunks with Claude on Bedrock."""
    context = "\n\n".join(
        f'<document index="{i}" source="{c["source"]}">\n{c["text"]}\n</document>'
        for i, c in enumerate(chunks, 1)
    )
    client = AnthropicBedrockMantle(aws_region=REGION)
    response = client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        output_config={"effort": "low"},  # less thinking, so more of the cap goes to the answer
        system=(
            "Answer using only the provided documents. Cite the source of each claim. "
            "If the documents do not contain the answer, say so instead of guessing. "
            "Be concise: answer in at most 150 words."
        ),
        messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
    )
    if response.stop_reason == "refusal":
        return "The model declined to answer this request."
    answer = "".join(b.text for b in response.content if b.type == "text")
    if response.stop_reason == "max_tokens":
        answer += "\n\n[truncated: hit the output limit]"
    return answer


# --------------------------------------------------------------------------- local RAG
def load_chunks(path: str, size: int = 1200, overlap: int = 200) -> list[Chunk]:
    """Extract PDF text page by page and split it into overlapping character windows."""
    from pypdf import PdfReader  # local mode only; not needed in Lambda

    chunks = []
    for page_no, page in enumerate(PdfReader(path).pages, 1):
        text = " ".join((page.extract_text() or "").split())
        for start in range(0, len(text), size - overlap):
            piece = text[start:start + size]
            if piece.strip():
                chunks.append({"source": f"{os.path.basename(path)} p.{page_no}", "text": piece})
    return chunks


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts with Titan Text Embeddings V2 (one call per text); vectors are unit length."""
    import numpy as np  # local mode only; not needed in Lambda

    runtime = boto3.client("bedrock-runtime", region_name=REGION)
    vectors = []
    for text in texts:
        resp = runtime.invoke_model(
            modelId=EMBED_MODEL,
            body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True}),
        )
        vectors.append(json.loads(resp["body"].read())["embedding"])
    return np.array(vectors)


def rag_local(question: str, k: int = 5) -> str:
    import numpy as np

    chunks = load_chunks(PDF_PATH)
    index = embed([c["text"] for c in chunks])  # cache this to disk for real use
    scores = index @ embed([question])[0]  # cosine similarity (vectors are normalized)
    top = [chunks[i] for i in np.argsort(scores)[::-1][:k]]
    return generate(question, top)


# --------------------------------------------------------------------------- managed RAG
def upload() -> None:
    key = os.path.basename(PDF_PATH)
    boto3.client("s3", region_name=REGION).upload_file(PDF_PATH, os.environ["S3_BUCKET"], key)
    print(f"Uploaded to s3://{os.environ['S3_BUCKET']}/{key}")


def ingest() -> None:
    """Sync the S3 data source into the vector store (run after adding/changing documents)."""
    job = boto3.client("bedrock-agent", region_name=REGION).start_ingestion_job(
        knowledgeBaseId=KB_ID, dataSourceId=os.environ["DATA_SOURCE_ID"]
    )["ingestionJob"]
    print(f"Started ingestion job {job['ingestionJobId']} ({job['status']})")


def rag_kb(question: str, k: int = 5) -> str:
    """Retrieve from the Knowledge Base, then generate with Claude."""
    resp = boto3.client("bedrock-agent-runtime", region_name=REGION).retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": question},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": k}},
    )
    chunks = [
        {
            "source": r["location"].get("s3Location", {}).get("uri", "unknown"),
            "text": r["content"]["text"],
        }
        for r in resp["retrievalResults"]
    ]
    return generate(question, chunks)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "upload":
        upload()
    elif mode == "ingest":
        ingest()
    elif mode in ("local", "kb") and len(sys.argv) > 2:
        question = " ".join(sys.argv[2:])
        print(rag_local(question) if mode == "local" else rag_kb(question))
    else:
        sys.exit(__doc__)
