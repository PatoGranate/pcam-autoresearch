"""Training and evaluation loops."""

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EvalResult:
    loss: float
    indices: np.ndarray   # patch index in the original split
    labels: np.ndarray    # true label, 0 or 1
    probs: np.ndarray     # predicted P(tumour)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    log_prefix: str = "",
) -> tuple[float, float]:
    """One pass over the training data. Returns (mean loss, accuracy).

    Prints a progress line roughly every 10% of the epoch, starting with `log_prefix`.
    """
    model.train()   # enables dropout and batch-norm statistic updates
    total_loss, correct, seen = 0.0, 0, 0
    n_batches = len(loader)
    report_at = {n_batches * k // 10 for k in range(1, 10)}   # batch numbers at ~10%, 20%, ... 90%
    start = time.time()

    for step, (images, labels, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += ((logits > 0) == labels.bool()).sum().item()   # logit > 0  <=>  prob > 0.5
        seen += len(labels)

        if step in report_at:
            elapsed = time.time() - start
            remaining = elapsed / step * (n_batches - step)
            print(f"{log_prefix}{step / n_batches:4.0%} ({step}/{n_batches} batches) | "
                  f"loss {total_loss / seen:.4f} acc {correct / seen:.4f} | "
                  f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s left")

    return total_loss / seen, correct / seen


@torch.no_grad()   # no gradients needed: faster and uses less memory
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> EvalResult:
    """Predict every patch in `loader` and return per-patch results."""
    model.eval()    # disables dropout; batch-norm uses its learned statistics
    total_loss, seen = 0.0, 0
    all_indices, all_labels, all_probs = [], [], []

    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True).float()

        logits = model(images)
        total_loss += loss_fn(logits, labels_dev).item() * len(labels)
        seen += len(labels)

        all_indices.append(indices.numpy())
        all_labels.append(labels.numpy())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())

    return EvalResult(
        loss=total_loss / seen,
        indices=np.concatenate(all_indices),
        labels=np.concatenate(all_labels),
        probs=np.concatenate(all_probs),
    )
