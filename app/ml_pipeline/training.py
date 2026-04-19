# backend/app/ml_pipeline/training.py
"""
MODEL TRAINING
===============
This module handles two types of training:

1. EMBEDDING FINE-TUNING: Improve retrieval quality by teaching the embedding
   model that certain questions should be close to certain document chunks.

2. CLASSIFIER TRAINING: Train a classifier on your domain data
   (e.g., categorise documents, detect intent, sentiment analysis).

We use scikit-learn for quick experiments and sentence-transformers
for embedding fine-tuning. In production, you might use PyTorch directly.
"""
import json
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from app.ml_pipeline.dataset import TrainingExample


class EmbeddingFineTuner:
    """
    Fine-tunes a sentence-transformer model on your domain data.

    WHY fine-tune embeddings?
    ==========================
    Generic embedding models (like text-embedding-3-small) work well for general text.
    But for domain-specific content (medical, legal, your company's docs), a fine-tuned
    model can improve retrieval accuracy by 10-30%.

    HOW it works:
    We train the model to make question embeddings CLOSER to their correct context
    embeddings, and FURTHER from unrelated contexts. This is called "contrastive learning".
    """

    def __init__(self, base_model: str = "all-MiniLM-L6-v2"):
        """
        all-MiniLM-L6-v2 is a small, fast model good for fine-tuning.
        In production, you might start with a larger model like:
        - all-mpnet-base-v2 (better quality, slower)
        - bge-large-en-v1.5 (state-of-the-art open-source)
        """
        self.model = SentenceTransformer(base_model)

    def prepare_training_data(
        self, examples: List[TrainingExample]
    ) -> List[InputExample]:
        """
        Converts our TrainingExamples into the format sentence-transformers expects.

        Each InputExample is a (question, context) pair.
        The model learns that these should have high similarity.
        """
        return [
            InputExample(texts=[ex.question, ex.context])
            for ex in examples
        ]

    def train(
        self,
        train_examples: List[TrainingExample],
        val_examples: List[TrainingExample],
        epochs: int = 3,
        batch_size: int = 16,
        output_path: str = "./fine_tuned_model",
    ) -> Dict:
        """
        Trains the embedding model.

        MultipleNegativesRankingLoss:
        - In each batch, the correct context is the "positive"
        - All OTHER contexts in the batch are "negatives"
        - The model learns to score positives higher than negatives
        - Efficient: you get N-1 negatives for free from the batch
        """
        train_data = self.prepare_training_data(train_examples)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

        # This loss function is specifically designed for retrieval tasks
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        # Train
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=int(len(train_dataloader) * 0.1),  # gradual learning rate increase
            output_path=output_path,
            show_progress_bar=True,
        )

        # Evaluate on validation set
        metrics = self.evaluate(val_examples)
        return metrics

    def evaluate(self, examples: List[TrainingExample]) -> Dict:
        """
        Evaluates retrieval quality using Mean Reciprocal Rank (MRR) and Recall@K.

        MRR: Average of 1/rank of the correct answer.
             MRR=1.0 means correct answer is always first. MRR=0.5 means usually 2nd.

        Recall@K: What fraction of correct answers appear in the top K results?
                  Recall@5 = 0.8 means 80% of questions find the right answer in top 5.
        """
        questions = [ex.question for ex in examples]
        contexts = [ex.context for ex in examples]

        # Embed all questions and contexts
        q_embeddings = self.model.encode(questions, convert_to_numpy=True)
        c_embeddings = self.model.encode(contexts, convert_to_numpy=True)

        # Compute similarity matrix
        similarities = np.dot(q_embeddings, c_embeddings.T)

        # Calculate metrics
        mrr_sum = 0
        recall_at_5 = 0

        for i in range(len(questions)):
            # Rank contexts by similarity to this question
            ranked_indices = np.argsort(similarities[i])[::-1]
            # Find where the correct context ranks
            correct_rank = np.where(ranked_indices == i)[0][0] + 1

            mrr_sum += 1.0 / correct_rank
            if correct_rank <= 5:
                recall_at_5 += 1

        n = len(questions)
        return {
            "mrr": mrr_sum / n,
            "recall_at_5": recall_at_5 / n,
            "num_examples": n,
        }