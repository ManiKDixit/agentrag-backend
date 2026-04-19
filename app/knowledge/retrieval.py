# backend/app/knowledge/retrieval.py
"""
RETRIEVAL: Finding the most relevant chunks for a given query.

This is the "R" in RAG. The quality of retrieval directly determines
the quality of the final answer. Bad retrieval = hallucinated answers.

Our retrieval pipeline:
1. Embed the user's question
2. Vector search in pgvector (approximate nearest neighbours)
3. (Optional) Rerank results for better precision
4. Return top-K chunks with their similarity scores
"""
from typing import List, Optional
from dataclasses import dataclass

from supabase import Client
from openai import OpenAI

from app.knowledge.embeddings import EmbeddingService


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its metadata."""
    id: str
    content: str
    document_id: str
    chunk_index: int
    similarity: float  # 0 to 1, where 1 = perfect match
    metadata: dict


class RetrievalService:
    def __init__(self, supabase: Client, openai_client: OpenAI):
        self.supabase = supabase
        self.embedding_service = EmbeddingService(openai_client)

    async def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
    ) -> List[RetrievedChunk]:
        """
        The core retrieval function.

        Parameters:
        - query: the user's question in natural language
        - user_id: ensures we only search THIS user's documents (privacy!)
        - top_k: how many chunks to return (5 is a good default)
        - similarity_threshold: minimum similarity to include (filters noise)

        SIMILARITY THRESHOLD explained:
        - 0.7+  = very relevant, nearly exact match
        - 0.5-0.7 = relevant, likely useful
        - 0.3-0.5 = somewhat relevant, might help
        - <0.3 = probably noise, discard
        """
        # Step 1: Embed the question
        query_embedding = await self.embedding_service.embed_query(query)

        # Step 2: Call our Supabase RPC function for vector search
        # This is the match_documents function we created in SQL earlier
        response = self.supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": top_k,
                "filter_user_id": user_id,
            },
        ).execute()

        # Step 3: Filter by similarity threshold and convert to dataclass
        chunks = []
        for row in response.data:
            if row["similarity"] >= similarity_threshold:
                chunks.append(
                    RetrievedChunk(
                        id=row["id"],
                        content=row["content"],
                        document_id=row["document_id"],
                        chunk_index=row["chunk_index"],
                        similarity=row["similarity"],
                        metadata=row.get("metadata", {}),
                    )
                )

        return chunks

    def format_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Formats retrieved chunks into a context string for the LLM.

        WHY format matters:
        The LLM needs to know which chunks are separate and how relevant they are.
        We include similarity scores so the LLM can weigh information appropriately.
        """
        if not chunks:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i} | Relevance: {chunk.similarity:.2f}]\n{chunk.content}"
            )
        return "\n\n---\n\n".join(context_parts)