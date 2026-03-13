"""FastAPI web server for the PDF RAG system."""

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRouter
from pydantic import BaseModel

from rag import PDFRag

app = FastAPI(title="PDF RAG")
api = APIRouter(prefix="/api")

_rag = None

def get_rag() -> PDFRag:
    global _rag
    if _rag is None:
        _rag = PDFRag()
    return _rag


class QueryRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()


@api.post("/ingest-file")
async def ingest_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    tmp_dir = Path(tempfile.mkdtemp(dir="/tmp"))
    named = tmp_dir / file.filename
    named.write_bytes(await file.read())

    try:
        count = get_rag().ingest(str(named))
        return {"message": f"Ingested '{file.filename}' — {count} chunks added."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        named.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


@api.post("/query")
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        answer = get_rag().query(req.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/sources")
async def sources():
    return {"sources": get_rag().list_sources()}


app.include_router(api)
