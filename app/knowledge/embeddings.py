# backend/app/knowledge/embeddings.py
"""
EMBEDDINGS: Converting text into numerical vectors.

KEY CONCEPT — What is an embedding?
====================================
An embedding is a list of numbers (a vector) that captures the MEANING of text.

"The cat sat on the mat"  →  [0.023, -0.541, 0.112, ..., 0.089]  (1536 numbers)
"A feline rested on a rug" →  [0.025, -0.538, 0.115, ..., 0.091]  (very similar!)
"Stock market crashed today" → [0.891, 0.234, -0.667, ..., 0.445]  (very different!)

These vectors live in a high-dimensional space where:
- SIMILAR meanings → vectors point in similar directions (high cosine similarity)
- DIFFERENT meanings → vectors point in different directions (low cosine similarity)

This is what makes semantic search possible: instead of keyword matching,
we find chunks whose MEANING is closest to the question's meaning.
"""
from typing import List
from openai import OpenAI

from app.config import get_settings


class EmbeddingService:
    def __init__(self, client: OpenAI):
        self.client = client
        self.settings = get_settings()


    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
            """
            Generates embeddings in batches to stay under OpenAI's token limit.
            Processes 100 chunks at a time instead of all at once.
            """
            batch_size = 100  # safe batch size to stay under token limits
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embeddings.create(
                    model=self.settings.embedding_model,
                    input=batch,
                )
                sorted_data = sorted(response.data, key=lambda x: x.index)
                all_embeddings.extend([item.embedding for item in sorted_data])

            return all_embeddings
    # async def embed_texts(self, texts: List[str]) -> List[List[float]]:
    #     """
    #     Generates embeddings for a batch of texts.

    #     WHY batch? OpenAI's API supports up to 2048 texts per request.
    #     Batching is ~10x faster than embedding one at a time.

    #     Returns a list of vectors, one per input text.
    #     """
    #     response = self.client.embeddings.create(
    #         model=self.settings.embedding_model,
    #         input=texts,
    #     )
    #     # Sort by index to maintain order (API doesn't guarantee order)
    #     sorted_data = sorted(response.data, key=lambda x: x.index)
    #     return [item.embedding for item in sorted_data]



    async def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string.
        Separated from embed_texts because some models use different
        prefixes for queries vs documents (e.g., "query: " vs "passage: ").
        """
        response = self.client.embeddings.create(
            model=self.settings.embedding_model,
            input=query,
        )
        return response.data[0].embedding