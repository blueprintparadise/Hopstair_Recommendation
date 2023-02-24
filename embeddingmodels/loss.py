import torch
from torch import nn


class CustomCrossEntropyLoss:

    r"""
    The most commonly found strategy is called in-batch negative sampling: for a specific observation
    in a batch we consider every other observations in this same batch as negatives. This is because a full
    evaluation will not be possible.

    Args:
        label_probs: precompute the relative abundance of each label for the weighting

    """

    def __init__(self, label_probs):
        self.label_probs = label_probs

    def __call__(self, true_labels, logits, training: bool = False):
        batch_size, nb_candidates = logits.shape

        if training:
            label_probs = torch.zeros(true_labels.shape)
            for label in true_labels:
                label_probs[true_labels == label] = self.label_probs[label] * torch.ones(true_labels.shape)
            logits -= torch.log(label_probs)

            true_labels = torch.range(0, batch_size)

        loss = nn.functional.cross_entropy(logits, true_labels)
        return torch.sum(loss)
