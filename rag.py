"""PDF RAG using ChromaDB + Claude."""

import os
import uuid
from pathlib import Path

import pdfplumber
import chromadb
from chromadb.utils import embedding_functions
import anthropic


CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between chunks
TOP_K = 5              # number of chunks to retrieve
MODEL = "claude-opus-4-6"


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
    """PDF Retrieval-Augmented Generation using ChromaDB and Claude."""

    def __init__(self, collection_name: str = "pdf_rag", persist_dir: str = None):
        if persist_dir is None:
            persist_dir = os.environ.get("CHROMA_DIR", "./chroma_db")
        self.client = anthropic.Anthropic()
        self.chroma = chromadb.PersistentClient(path=persist_dir)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
        )

    def ingest(self, pdf_path: str) -> int:
        """Parse a PDF and store its chunks in ChromaDB. Returns number of chunks added."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"Extracting text from {path.name}...")
        text = extract_text(pdf_path)
        chunks = chunk_text(text)

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": path.name, "chunk_index": i} for i in range(len(chunks))]

        print(f"Storing {len(chunks)} chunks in ChromaDB...")
        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        print(f"Done. Ingested {len(chunks)} chunks from '{path.name}'.")
        return len(chunks)

    def query(self, question: str, top_k: int = TOP_K) -> str:
        """Retrieve relevant chunks and ask Claude to answer the question."""
        results = self.collection.query(query_texts=[question], n_results=top_k)
        chunks = results["documents"][0]
        sources = [m["source"] for m in results["metadatas"][0]]

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
        results = self.collection.get(include=["metadatas"])
        sources = {m["source"] for m in results["metadatas"]}
        return sorted(sources)
