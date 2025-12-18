"""
NLP model definitions and wrappers.
"""
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


def get_distilbert_sst2(num_labels=2):
    """
    Get DistilBERT model for SST-2 sentiment analysis.

    Args:
        num_labels: Number of output classes (2 for binary classification)

    Returns:
        DistilBERT model
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels
    )
    return model
