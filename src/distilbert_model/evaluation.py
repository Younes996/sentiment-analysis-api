from pathlib import Path
import argparse
import json

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    PreTrainedTokenizerBase
)
from pandas import DataFrame

from src.distilbert_model.data import encode_batch

from src.sentiment_model.config import TRAIN_PATH, PARTITIONS_PATH
from src.sentiment_model.data import load_train_validation


def count_confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Count predictions with true labels as rows and predicted labels as columns."""
    predictions = logits.argmax(dim=1)
    confusion_matrix = torch.zeros((2, 2), dtype=torch.long)

    confusion_matrix[0, 0] = ((labels == 0) & (predictions == 0)).sum()
    confusion_matrix[0, 1] = ((labels == 0) & (predictions == 1)).sum()
    confusion_matrix[1, 0] = ((labels == 1) & (predictions == 0)).sum()
    confusion_matrix[1, 1] = ((labels == 1) & (predictions == 1)).sum()

    return confusion_matrix


def evaluate_classifier(
    model: DistilBertForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    examples: DataFrame,
    batch_size: int,
) -> tuple[float, torch.Tensor]:
    """Compute classification accuracy and the confusion matrix across all examples."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if examples.empty:
        raise ValueError("Evaluation requires at least one example.")

    model.eval()
    confusion_matrix = torch.zeros((2, 2), dtype=torch.long)
    total_batches = (len(examples) + batch_size - 1) // batch_size
    log_interval = 100

    with torch.inference_mode():
        for batch_number, start in enumerate(
            range(0, len(examples), batch_size),
            start=1,
        ):
            batch_examples = examples.iloc[start:start + batch_size]
            batch_inputs, batch_labels = encode_batch(tokenizer, batch_examples)
            batch_logits = model(**batch_inputs).logits
            confusion_matrix += count_confusion_matrix(batch_logits, batch_labels)

            if batch_number % log_interval == 0 or batch_number == total_batches:
                print(f"Evaluation batch {batch_number}/{total_batches}", flush=True)

    correct_predictions = confusion_matrix.diagonal().sum().item()
    accuracy = correct_predictions / len(examples)

    return accuracy, confusion_matrix


def compute_positive_precision(confusion_matrix: torch.Tensor) -> float:
    """Return precision for the positive class."""
    TP = confusion_matrix[1, 1].item()
    FP = confusion_matrix[0, 1].item()

    if TP + FP == 0:
        return 0.0
    return TP / (TP + FP)


def compute_positive_recall(confusion_matrix: torch.Tensor) -> float:
    """Return recall for the positive class."""
    TP = confusion_matrix[1, 1].item()
    FN = confusion_matrix[1, 0].item()

    if TP + FN == 0:
        return 0.0
    return TP / (TP + FN)


def compute_f1(precision: float, recall: float) -> float:
    """Return the harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    """Evaluate a local DistilBERT checkpoint on a frozen IMDb split."""
    parser = argparse.ArgumentParser(
        description="Evaluate a local DistilBERT checkpoint."
    )
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Path to the saved model and tokenizer directory.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
    )
    args = parser.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint_dir,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_dir,
        local_files_only=True
    )
    train_df, val_df = load_train_validation(TRAIN_PATH, PARTITIONS_PATH)

    examples = train_df if args.split == "train" else val_df
    accuracy, confusion_matrix = evaluate_classifier(
        model, tokenizer, examples, batch_size=2
    )

    precision = compute_positive_precision(confusion_matrix)
    recall = compute_positive_recall(confusion_matrix)
    f1_score = compute_f1(precision, recall)

    print(f"{args.split} set:\n")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Positive-class precision: {precision:.2%}")
    print(f"Positive-class recall: {recall:.2%}")
    print(f"Positive-class F1: {f1_score:.2%}")
    print("Confusion matrix (rows: true labels, columns: predicted labels):")
    print(confusion_matrix)

    metrics = {
        f"{args.split}_size": len(examples),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": confusion_matrix.tolist(),
    }

    metrics_path = args.checkpoint_dir.parent / f"{args.split}_metrics.json"
    with open(metrics_path, mode="w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)


if __name__ == "__main__":
    main()
