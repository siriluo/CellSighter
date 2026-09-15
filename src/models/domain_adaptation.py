import math

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversal(Function):
    """Identity in the forward pass, gradient sign flip in the backward pass."""

    @staticmethod
    def forward(ctx, x, lambda_value):
        ctx.lambda_value = lambda_value
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_value * grad_output, None


def reverse_gradient(x, lambda_value=1.0):
    return GradientReversal.apply(x, lambda_value)


class DomainDiscriminator(nn.Module):
    """
    Small domain classifier for DANN-style adaptation.

    It sees encoder features, not projection-head features, so the SupCon space can
    stay focused on class structure while the encoder learns stain/dataset invariance.
    """

    def __init__(self, in_dim, hidden_dim=256, num_domains=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, features, lambda_value=1.0):
        return self.net(reverse_gradient(features, lambda_value))


def dann_lambda(epoch, batch_idx, batches_per_epoch, max_epochs, max_lambda=1.0, warmup_epochs=0):
    """
    Smooth DANN schedule from the original GRL paper, with optional warmup.

    The warmup keeps early SupCon training from being dominated by the adversary.
    """
    if max_lambda <= 0:
        return 0.0

    progress_step = max(0, (epoch - 1) * batches_per_epoch + batch_idx)
    warmup_steps = max(0, warmup_epochs * batches_per_epoch)
    total_steps = max(1, max_epochs * batches_per_epoch - warmup_steps)

    if progress_step < warmup_steps:
        return 0.0

    p = min(1.0, (progress_step - warmup_steps) / total_steps)
    return float(max_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0))
