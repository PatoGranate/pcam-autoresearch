"""Figures saved into each run folder."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # write image files without opening a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from pcam.data import PCamPatches
from pcam.engine import EvalResult

CLASS_NAMES = ("normal", "tumour")


def plot_curves(history: list[dict], path: Path) -> None:
    """Loss and accuracy per epoch, train vs validation."""
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], "o-", label="val")
    ax_loss.set(title="Loss", xlabel="epoch", ylabel="BCE loss")

    ax_acc.plot(epochs, [h["train_acc"] for h in history], "o-", label="train")
    ax_acc.plot(epochs, [h["val_acc"] for h in history], "o-", label="val")
    ax_acc.set(title="Accuracy", xlabel="epoch", ylabel="accuracy")

    for ax in (ax_loss, ax_acc):
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xticks(epochs)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm: dict, path: Path) -> None:
    """2x2 grid of counts; rows = true class, columns = predicted class."""
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    row_share = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            colour = "white" if row_share[i, j] > 0.5 else "black"
            ax.text(j, i, f"{matrix[i, j]:,}\n({row_share[i, j]:.1%} of row)",
                    ha="center", va="center", color=colour)
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set(xlabel="predicted", ylabel="true", title="Test confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_worst_mistakes(dataset: PCamPatches, result: EvalResult, path: Path, n: int = 16) -> None:
    """Grid of the wrong predictions the model was most confident about.

    The dashed box marks the central 32x32 region: the label only says whether
    *that* region contains tumour. Confident "mistakes" are worth inspecting:
    some are model errors, some may be labelling errors (Phase 2's topic).
    """
    preds = (result.probs >= 0.5).astype(int)
    wrong = np.flatnonzero(preds != result.labels)
    # How sure the model was of its (wrong) answer: P(predicted class)
    confidence = np.where(preds == 1, result.probs, 1 - result.probs)
    worst = wrong[np.argsort(-confidence[wrong])][:n]

    cols = 4
    rows = max(1, int(np.ceil(len(worst) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9))
    for ax in np.ravel(axes):
        ax.axis("off")

    for ax, i in zip(np.ravel(axes), worst):
        index = int(result.indices[i])
        ax.imshow(dataset.load_image(index))    # raw image, without normalisation
        ax.add_patch(Rectangle((32, 32), 32, 32, fill=False, edgecolor="yellow",
                               linestyle="--", linewidth=1.2))
        ax.set_title(f"#{index}  true: {CLASS_NAMES[result.labels[i]]}\n"
                     f"P(tumour) = {result.probs[i]:.3f}", fontsize=8)

    if len(worst) == 0:
        fig.suptitle("No mistakes on the test set")
    else:
        fig.suptitle("Most confidently wrong test patches (dashed: labelled region)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
