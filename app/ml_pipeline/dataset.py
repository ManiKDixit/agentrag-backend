# backend/app/ml_pipeline/dataset.py
"""
ML PIPELINE — DATASET PREPARATION
===================================
Before you can train any model, you need clean, structured data.

This module handles:
1. Extracting training data from your knowledge base
2. Creating question-answer pairs (for fine-tuning)
3. Generating synthetic training data using LLMs
4. Train/validation/test splitting
"""
import json
import random
from typing import List, Tuple
from dataclasses import dataclass

from openai import OpenAI
from supabase import Client


@dataclass
class TrainingExample:
    """A single training example for fine-tuning."""
    question: str
    context: str     # the chunk that contains the answer
    answer: str      # the expected answer
    metadata: dict


class DatasetBuilder:
    """
    Builds training datasets from your knowledge base.

    KEY CONCEPT — Synthetic Data Generation
    =========================================
    You rarely have enough labelled data to train models.
    Solution: use a powerful LLM (GPT-4) to GENERATE training data
    from your existing documents. This is called "synthetic data generation"
    and it's a standard industry practice.

    Flow:
    1. Take a document chunk
    2. Ask GPT-4: "Generate 3 questions that this text answers"
    3. Ask GPT-4: "Answer each question using ONLY this text"
    4. Now you have (question, context, answer) triples for training
    """

    def __init__(self, openai_client: OpenAI, supabase: Client):
        self.client = openai_client
        self.supabase = supabase

    async def generate_qa_pairs(
        self, chunk_content: str, num_pairs: int = 3
    ) -> List[dict]:
        """
        Uses GPT-4 to generate question-answer pairs from a text chunk.
        These pairs can be used to:
        - Fine-tune a smaller model
        - Train a custom embedding model
        - Evaluate RAG accuracy
        """
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a dataset generation assistant. Given a text passage,
generate question-answer pairs where:
- Questions are diverse (factual, inferential, comparative)
- Answers are grounded ONLY in the provided text
- Output valid JSON array of objects with "question" and "answer" fields""",
                },
                {
                    "role": "user",
                    "content": f"""Generate {num_pairs} question-answer pairs from this text:

{chunk_content}

Output JSON array:""",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.7,  # some creativity for diverse questions
        )

        result = json.loads(response.choices[0].message.content)
        return result.get("pairs", result.get("qa_pairs", []))

    async def build_dataset(
        self,
        user_id: str,
        max_chunks: int = 100,
        pairs_per_chunk: int = 3,
    ) -> List[TrainingExample]:
        """
        Builds a complete training dataset from a user's knowledge base.

        Steps:
        1. Fetch chunks from the database
        2. Generate QA pairs for each chunk
        3. Structure into TrainingExamples
        """
        # Fetch chunks
        response = (
            self.supabase.table("document_chunks")
            .select("content, document_id, metadata")
            .eq("user_id", user_id)
            .limit(max_chunks)
            .execute()
        )

        examples = []
        for chunk in response.data:
            qa_pairs = await self.generate_qa_pairs(
                chunk["content"], pairs_per_chunk
            )
            for pair in qa_pairs:
                examples.append(
                    TrainingExample(
                        question=pair["question"],
                        context=chunk["content"],
                        answer=pair["answer"],
                        metadata={"document_id": chunk["document_id"]},
                    )
                )

        return examples

    def train_val_test_split(
        self,
        examples: List[TrainingExample],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> Tuple[List, List, List]:
        """
        Splits data into train/validation/test sets.

        WHY three sets?
        - Train: model learns from this
        - Validation: used during training to detect overfitting
        - Test: final evaluation, NEVER seen during training

        If you only use train/test, you risk overfitting to the test set
        by tuning hyperparameters based on test performance.
        """
        random.shuffle(examples)
        n = len(examples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return (
            examples[:train_end],
            examples[train_end:val_end],
            examples[val_end:],
        )