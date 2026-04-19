# backend/app/ml_pipeline/evaluation.py
"""
EVALUATION MODULE
==================
Evaluating RAG systems requires different metrics than traditional ML.

We measure:
1. RETRIEVAL quality: Are we finding the right chunks?
2. GENERATION quality: Is the final answer correct and well-grounded?
3. END-TO-END quality: Does the whole pipeline work?
"""
from typing import List, Dict
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class EvalResult:
    question: str
    expected_answer: str
    actual_answer: str
    retrieval_score: float  # how relevant were the retrieved chunks?
    faithfulness: float     # is the answer grounded in the retrieved chunks?
    correctness: float      # does the answer match the expected answer?


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using LLM-as-judge.

    WHY LLM-as-judge?
    Traditional metrics (BLEU, ROUGE) compare word overlap.
    But "Paris is France's capital" and "The capital of France is Paris"
    have low BLEU but are both correct. LLMs understand semantic equivalence.
    """

    def __init__(self, openai_client: OpenAI):
        self.client = openai_client

    async def evaluate_faithfulness(
        self, answer: str, context: str
    ) -> float:
        """
        Checks: Is the answer GROUNDED in the provided context?

        Score:
        1.0 = Fully grounded, every claim is in the context
        0.5 = Partially grounded, some claims not in context
        0.0 = Not grounded, answer contradicts or fabricates
        """
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You evaluate whether an answer is faithful to the provided context.
Score from 0.0 to 1.0:
- 1.0: Every claim in the answer is supported by the context
- 0.5: Some claims are supported, others are not
- 0.0: The answer contradicts or fabricates information not in the context
Return ONLY a JSON object: {"score": <float>, "reasoning": "<explanation>"}""",
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nAnswer: {answer}",
                },
            ],
            response_format={"type": "json_object"},
        )

        import json
        result = json.loads(response.choices[0].message.content)
        return result["score"]

    async def evaluate_correctness(
        self, answer: str, expected_answer: str
    ) -> float:
        """
        Checks: Does the answer convey the same information as the expected answer?
        Uses semantic comparison, not exact string matching.
        """
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You evaluate whether two answers convey the same information.
Score from 0.0 to 1.0:
- 1.0: Semantically equivalent
- 0.5: Partially correct
- 0.0: Wrong or contradictory
Return ONLY a JSON object: {"score": <float>, "reasoning": "<explanation>"}""",
                },
                {
                    "role": "user",
                    "content": f"Expected: {expected_answer}\n\nActual: {answer}",
                },
            ],
            response_format={"type": "json_object"},
        )

        import json
        result = json.loads(response.choices[0].message.content)
        return result["score"]

    async def run_evaluation(
        self,
        test_cases: List[Dict],
        rag_pipeline_fn,  # function that takes a question and returns (answer, context)
    ) -> Dict:
        """
        Runs full evaluation across a test set.

        Returns aggregate metrics:
        - Average faithfulness
        - Average correctness
        - Score distribution
        """
        results = []

        for case in test_cases:
            answer, context = await rag_pipeline_fn(case["question"])

            faithfulness = await self.evaluate_faithfulness(answer, context)
            correctness = await self.evaluate_correctness(
                answer, case["expected_answer"]
            )

            results.append(EvalResult(
                question=case["question"],
                expected_answer=case["expected_answer"],
                actual_answer=answer,
                retrieval_score=0.0,  # TODO: implement
                faithfulness=faithfulness,
                correctness=correctness,
            ))

        # Aggregate
        n = len(results)
        return {
            "num_examples": n,
            "avg_faithfulness": sum(r.faithfulness for r in results) / n,
            "avg_correctness": sum(r.correctness for r in results) / n,
            "results": [
                {
                    "question": r.question,
                    "faithfulness": r.faithfulness,
                    "correctness": r.correctness,
                }
                for r in results
            ],
        }