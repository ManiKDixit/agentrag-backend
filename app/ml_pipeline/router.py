# backend/app/ml_pipeline/router.py
"""
API endpoints for the ML pipeline.
These let you trigger dataset generation, training, and evaluation from the frontend.
"""
from fastapi import APIRouter, Depends, BackgroundTasks
from pydantic import BaseModel
from supabase import Client
from openai import OpenAI

from app.dependencies import get_supabase_admin, get_openai_client, get_current_user
from app.ml_pipeline.dataset import DatasetBuilder

router = APIRouter()


class DatasetRequest(BaseModel):
    max_chunks: int = 50
    pairs_per_chunk: int = 3


@router.post("/generate-dataset")
async def generate_dataset(
    request: DatasetRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin),
    openai_client: OpenAI = Depends(get_openai_client),
):
    """
    Generates a training dataset from the user's knowledge base.

    Uses BackgroundTasks because this can take minutes for large knowledge bases.
    The frontend can poll a status endpoint to check progress.
    """
    builder = DatasetBuilder(openai_client, supabase)

    # For a real app, run this in background and track progress
    examples = await builder.build_dataset(
        user_id=user["id"],
        max_chunks=request.max_chunks,
        pairs_per_chunk=request.pairs_per_chunk,
    )

    # Split into train/val/test
    train, val, test = builder.train_val_test_split(examples)

    return {
        "total_examples": len(examples),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "sample": {
            "question": examples[0].question if examples else "",
            "answer": examples[0].answer if examples else "",
        },
    }


@router.post("/evaluate")
async def evaluate_rag(
    user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin),
    openai_client: OpenAI = Depends(get_openai_client),
):
    """Runs evaluation on the RAG pipeline."""
    from app.ml_pipeline.evaluation import RAGEvaluator
    from app.knowledge.retrieval import RetrievalService

    evaluator = RAGEvaluator(openai_client)
    retrieval = RetrievalService(supabase, openai_client)

    async def rag_fn(question: str):
        chunks = await retrieval.retrieve(question, user["id"], top_k=3)
        context = "\n".join([c.content for c in chunks])
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Answer based on this context:\n{context}"},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content, context

    # Generate test cases with more chunks for reliability
    builder = DatasetBuilder(openai_client, supabase)
    examples = await builder.build_dataset(user["id"], max_chunks=10, pairs_per_chunk=2)

    if not examples:
        return {
            "num_examples": 0,
            "avg_faithfulness": 0.0,
            "avg_correctness": 0.0,
            "results": [],
        }

    # Use up to 5 examples for evaluation
    test_cases = [
        {"question": ex.question, "expected_answer": ex.answer}
        for ex in examples[:5]
    ]

    try:
        results = await evaluator.run_evaluation(test_cases, rag_fn)
        return results
    except Exception as e:
        return {
            "num_examples": len(test_cases),
            "avg_faithfulness": 0.0,
            "avg_correctness": 0.0,
            "results": [],
            "error": str(e),
        }