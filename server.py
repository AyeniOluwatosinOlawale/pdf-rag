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
rag = PDFRag()
api = APIRouter(prefix="/api")


class QueryRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()


@api.post("/ingest-file")
async def ingest_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    tmp_dir = Path(tempfile.mkdtemp())
    named = tmp_dir / file.filename
    named.write_bytes(await file.read())

    try:
        count = rag.ingest(str(named))
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
        answer = rag.query(req.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/sources")
async def sources():
    return {"sources": rag.list_sources()}


app.include_router(api)
