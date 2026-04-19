# backend/app/knowledge/router.py
"""
API endpoints for managing the knowledge base.
"""
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from supabase import Client
from openai import OpenAI

from app.dependencies import get_supabase_admin, get_openai_client, get_current_user
from app.knowledge.ingestion import parse_document, TextChunker
from app.knowledge.embeddings import EmbeddingService

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin),
    openai_client: OpenAI = Depends(get_openai_client),
):
    """
    Full ingestion pipeline for a single document.

    Flow: Upload → Parse → Chunk → Embed → Store

    This is where all the pieces come together. A single API call triggers
    the entire pipeline. In production, you'd make this async with a task queue
    (Celery, etc.) because embedding large docs can take minutes.
    """
    # 1. Read file bytes
    file_bytes = await file.read()

    # 2. Parse document to text
    try:
        text = parse_document(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 3. Chunk the text
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_text(text)

    if not chunks:
        raise HTTPException(status_code=400, detail="No text content found in document")

    # 4. Generate embeddings for ALL chunks in one batch
    embedding_service = EmbeddingService(openai_client)
    embeddings = await embedding_service.embed_texts(chunks)

    # 5. Store document metadata
    doc_response = supabase.table("documents").insert({
        "user_id": user["id"],
        "title": file.filename,
        "source_type": file.filename.split(".")[-1],
        "metadata": {"chunk_count": len(chunks), "char_count": len(text)},
    }).execute()

    document_id = doc_response.data[0]["id"]

    # 6. Store chunks with their embeddings
    chunk_records = [
        {
            "document_id": document_id,
            "user_id": user["id"],
            "content": chunk,
            "chunk_index": i,
            "embedding": embedding,
            "metadata": {"source": file.filename},
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    # Batch insert (Supabase handles this efficiently)
    supabase.table("document_chunks").insert(chunk_records).execute()

    return {
        "document_id": document_id,
        "chunks_created": len(chunks),
        "message": f"Successfully ingested '{file.filename}'",
    }


@router.get("/documents")
async def list_documents(
    user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin),
):
    """Lists all documents for the authenticated user."""
    response = (
        supabase.table("documents")
        .select("*")
        .eq("user_id", user["id"])
        .order("created_at", desc=True)
        .execute()
    )
    return response.data


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin),
):
    """
    Deletes a document and all its chunks.
    The ON DELETE CASCADE in our SQL schema means deleting the document
    automatically deletes all related chunks — no orphaned data.
    """
    # Verify ownership
    doc = (
        supabase.table("documents")
        .select("id")
        .eq("id", document_id)
        .eq("user_id", user["id"])
        .single()
        .execute()
    )
    if not doc.data:
        raise HTTPException(status_code=404, detail="Document not found")

    supabase.table("documents").delete().eq("id", document_id).execute()
    return {"message": "Document deleted"}