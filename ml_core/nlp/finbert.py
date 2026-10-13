"""FinBERT (ProsusAI/finbert) sentence-level financial sentiment."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Sequence

MODEL_NAME = "ProsusAI/finbert"


@dataclass(frozen=True)
class SentimentScore:
    positive: float
    negative: float
    neutral: float

    @property
    def score(self) -> float:
        """Net sentiment in [-1, 1]."""
        return self.positive - self.negative


class FinBert:
    _instance: "FinBert | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()
        labels = {index: label.lower() for index, label in self.model.config.id2label.items()}
        self.index = {label: index for index, label in labels.items()}
        missing = {"positive", "negative", "neutral"} - set(self.index)
        if missing:
            raise RuntimeError(f"Unexpected FinBERT labels: {labels}")

    @classmethod
    def get(cls) -> "FinBert":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def score(self, texts: Sequence[str], batch_size: int = 16, max_length: int = 256) -> list[SentimentScore]:
        results: list[SentimentScore] = []
        for start in range(0, len(texts), batch_size):
            batch = [text_value or "" for text_value in texts[start:start + batch_size]]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            with self.torch.no_grad():
                probabilities = self.torch.softmax(self.model(**encoded).logits, dim=-1).cpu().numpy()
            for row in probabilities:
                results.append(
                    SentimentScore(
                        positive=float(row[self.index["positive"]]),
                        negative=float(row[self.index["negative"]]),
                        neutral=float(row[self.index["neutral"]]),
                    )
                )
        return results
