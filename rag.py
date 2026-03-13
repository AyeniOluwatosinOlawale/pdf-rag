"""PDF RAG using Qdrant + Claude."""

import os
import uuid
from pathlib import Path

import pdfplumber
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import anthropic


CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between chunks
TOP_K = 5              # number of chunks to retrieve
MODEL = "claude-opus-4-6"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384      # output dimension for all-MiniLM-L6-v2

# Cache fastembed models in /tmp so Vercel's read-only filesystem isn't a problem
os.environ["FASTEMBED_CACHE_PATH"] = "/tmp/fastembed"

# Module-level singleton — model is downloaded once per container lifetime
_embed_model: "TextEmbedding | None" = None

def get_embed_model() -> "TextEmbedding":
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding(EMBEDDING_MODEL)
    return _embed_model


def extract_text(pdf_path: str) -> str:
    """Extract full text from a PDF file."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


class PDFRag:
    """PDF Retrieval-Augmented Generation using Qdrant and Claude."""

    def __init__(self, collection_name: str = "pdf_rag"):
        self.client = anthropic.Anthropic()
        self.collection_name = collection_name

        # Qdrant Cloud when QDRANT_URL is set, local file storage otherwise
        qdrant_url = os.environ.get("QDRANT_URL")
        if qdrant_url:
            self.qdrant = QdrantClient(
                url=qdrant_url,
                api_key=os.environ.get("QDRANT_API_KEY"),
            )
        else:
            persist_dir = os.environ.get("QDRANT_DIR", "./qdrant_db")
            self.qdrant = QdrantClient(path=persist_dir)

        existing = {c.name for c in self.qdrant.get_collections().collections}
        if collection_name not in existing:
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    def ingest(self, pdf_path: str) -> int:
        """Parse a PDF and store its chunks in Qdrant. Returns number of chunks added."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"Extracting text from {path.name}...")
        text = extract_text(pdf_path)
        chunks = chunk_text(text)

        print(f"Embedding {len(chunks)} chunks...")
        vectors = [v.tolist() for v in get_embed_model().embed(chunks)]

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"source": path.name, "chunk_index": i, "text": chunk},
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

        print(f"Storing {len(chunks)} chunks in Qdrant...")
        self.qdrant.upsert(collection_name=self.collection_name, points=points)
        print(f"Done. Ingested {len(chunks)} chunks from '{path.name}'.")
        return len(chunks)

    def query(self, question: str, top_k: int = TOP_K) -> str:
        """Retrieve relevant chunks and ask Claude to answer the question."""
        query_vector = next(get_embed_model().embed([question])).tolist()
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

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
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Context from documents:\n\n{context}\n\n"
                        f"Question: {question}"
                    ),
                }
            ],
        )

        for block in response.content:
            if block.type == "text":
                return block.text

        return ""

    def list_sources(self) -> list[str]:
        """Return the unique PDF sources currently ingested."""
        points, _ = self.qdrant.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            limit=10000,
        )
        sources = {point.payload["source"] for point in points}
        return sorted(sources)
