"""PDF RAG using Qdrant + Claude."""

import os
import uuid
from pathlib import Path

import requests
import pdfplumber
from docx import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import anthropic


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
MODEL = "claude-opus-4-6"
VECTOR_SIZE = 384
COHERE_API_URL = "https://api.cohere.com/v2/embed"
COHERE_MODEL = "embed-english-light-v3.0"  # 384-dim, matches VECTOR_SIZE


def embed(texts: list[str], input_type: str = "search_document", batch_size: int = 96) -> list[list[float]]:
    """Embed texts via Cohere API in batches."""
    headers = {
        "Authorization": f"Bearer {os.environ['COHERE_API_KEY']}",
        "Content-Type": "application/json",
    }
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            COHERE_API_URL,
            headers=headers,
            json={
                "model": COHERE_MODEL,
                "input_type": input_type,
                "texts": batch,
                "embedding_types": ["float"],
            },
            timeout=30,
        )
        resp.raise_for_status()
        all_embeddings.extend(resp.json()["embeddings"]["float"])
    return all_embeddings


def extract_text(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".docx":
        doc = Document(path)
        return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())
    # default: PDF
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


class PDFRag:
    """PDF Retrieval-Augmented Generation using Qdrant and Claude."""

    def __init__(self, collection_name: str = "pdf_rag"):
        self.client = anthropic.Anthropic()
        self.collection_name = collection_name

        qdrant_url = os.environ.get("QDRANT_URL")
        if qdrant_url:
            self.qdrant = QdrantClient(
                url=qdrant_url,
                api_key=os.environ.get("QDRANT_API_KEY"),
            )
        else:
            self.qdrant = QdrantClient(path=os.environ.get("QDRANT_DIR", "./qdrant_db"))

        existing = {c.name for c in self.qdrant.get_collections().collections}
        if collection_name not in existing:
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    def ingest(self, pdf_path: str) -> int:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"Extracting text from {path.name}...")
        text = extract_text(pdf_path)
        chunks = chunk_text(text)

        print(f"Embedding {len(chunks)} chunks...")
        vectors = embed(chunks)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"source": path.name, "chunk_index": i, "text": chunk},
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

        print(f"Storing {len(points)} chunks in Qdrant...")
        self.qdrant.upsert(collection_name=self.collection_name, points=points)
        print(f"Done. Ingested {len(points)} chunks from '{path.name}'.")
        return len(points)

    def query(self, question: str, top_k: int = TOP_K) -> str:
        query_vector = embed([question], input_type="search_query")[0]
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        chunks = [r.payload["text"] for r in results]
        sources = [r.payload["source"] for r in results]

        context = "\n\n---\n\n".join(
            f"[Source: {src}]\n{chunk}" for src, chunk in zip(sources, chunks)
        )

        system = (
            "You are a helpful assistant that answers questions based strictly on the "
            "provided document excerpts. If the answer is not in the excerpts, say so. "
            "Cite which source(s) your answer comes from."
        )

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=1024,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": f"Context from documents:\n\n{context}\n\nQuestion: {question}"}],
        )

        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    def list_sources(self) -> list[str]:
        points, _ = self.qdrant.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            limit=10000,
        )
        return sorted({point.payload["source"] for point in points})
