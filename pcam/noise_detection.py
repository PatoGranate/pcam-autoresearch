"""Find suspicious training labels from out-of-fold predictions (confident learning).

Idea: a model that has *seen* a patch can memorise its label, even a wrong one.
So we split the training patches into K folds, train K models, and let each
model predict only the fold it was NOT trained on. A label that such an
"unbiased" model confidently disagrees with is suspicious.
"""

import copy

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader

from pcam.config import Config
from pcam.data import PCamPatches, build_transform
from pcam.engine import evaluate, train_one_epoch
from pcam.metrics import compute_metrics
from pcam.models import build_model
from pcam.utils import seed_worker, set_seed


def _loader(ds: PCamPatches, positions: np.ndarray, cfg: Config, device: torch.device,
            train: bool, seed: int) -> DataLoader:
    """DataLoader over the patches at `positions` (positions within ds.indices).

    A shallow copy shares the label array and file path but has its own index list,
    and held-out folds get the plain (no-augmentation) transform.
    """
    part = copy.copy(ds)
    part.indices = ds.indices[positions]
    part.transform = build_transform(train=train)
    workers = cfg.data.num_workers
    return DataLoader(
        part, batch_size=cfg.training.batch_size, shuffle=train, num_workers=workers,
        pin_memory=(device.type == "cuda"), persistent_workers=workers > 0,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed),
    )


def out_of_fold_probs(train_ds: PCamPatches, cfg: Config, device: torch.device) -> tuple[np.ndarray, list[dict]]:
    """P(tumour) for every train patch, predicted by a model that never trained on it.

    Returns (probs aligned with train_ds.indices, one summary dict per fold).
    """
    given = train_ds.labels[train_ds.indices]          # the (possibly noisy) labels we are judging
    probs = np.zeros(len(given))
    folds = []
    splitter = StratifiedKFold(cfg.detection.folds, shuffle=True, random_state=cfg.seed)
    loss_fn = nn.BCEWithLogitsLoss()

    for k, (fit_pos, held_pos) in enumerate(splitter.split(np.zeros(len(given)), given), start=1):
        set_seed(cfg.seed + k)
        model = build_model(cfg.model).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                                      weight_decay=cfg.training.weight_decay)
        fit_loader = _loader(train_ds, fit_pos, cfg, device, train=True, seed=cfg.seed + k)
        held_loader = _loader(train_ds, held_pos, cfg, device, train=False, seed=cfg.seed + k)

        for epoch in range(1, cfg.training.epochs + 1):
            loss, acc = train_one_epoch(model, fit_loader, optimizer, loss_fn, device)
            print(f"  fold {k}/{cfg.detection.folds} epoch {epoch}/{cfg.training.epochs} | "
                  f"train loss {loss:.4f} acc {acc:.4f}")

        result = evaluate(model, held_loader, loss_fn, device)
        probs[held_pos] = result.probs
        # Held-out quality against the labels we have (noisy), as a sanity check on the fold model.
        folds.append({"fold": k, "n_held_out": int(len(held_pos)),
                      "held_out_auc_vs_given_labels": compute_metrics(result.labels, result.probs)["auc"]})
        print(f"Fold {k}/{cfg.detection.folds} done | held-out AUC vs given labels "
              f"{folds[-1]['held_out_auc_vs_given_labels']:.4f}")
        del model, optimizer, fit_loader, held_loader   # free GPU/worker memory before the next fold
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return probs, folds


def confident_learning_flags(given: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Boolean mask: True where the label looks wrong (binary confident learning).

    Per-class threshold t_c = average P(class c) over patches *labelled* c, i.e. how
    confident the models typically are on that class. A patch is a candidate for
    class c if P(c) >= t_c. A patch is flagged when its best candidate class differs
    from its given label. Patches that clear neither threshold are left alone.
    """
    p = np.stack([1 - probs, probs], axis=1)                      # columns: P(normal), P(tumour)
    thresholds = np.array([p[given == c, c].mean() for c in (0, 1)])
    candidate = p >= thresholds                                    # (n, 2) bool
    best = np.where(candidate, p, -np.inf).argmax(axis=1)          # most probable candidate class
    return candidate.any(axis=1) & (best != given)


def suspicion_score(given: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Higher = more suspicious: 1 - P(given label). Used for ranking."""
    return 1 - np.where(given == 1, probs, 1 - probs)


def score_detector(is_flipped: np.ndarray, score: np.ndarray, flagged: np.ndarray) -> dict:
    """Compare a detector with the known flips: set metrics for `flagged`, ranking metrics for `score`."""
    n_flipped = int(is_flipped.sum())
    out = {"n_samples": int(len(is_flipped)), "n_flipped": n_flipped,
           "n_flagged": int(flagged.sum()), "flagged_fraction": float(flagged.mean())}
    if n_flipped == 0:   # clean labels: nothing to find, so only the flag count is meaningful
        return out
    tp = int((flagged & is_flipped).sum())
    out["true_positives"] = tp
    out["precision"] = tp / max(1, out["n_flagged"])
    out["recall"] = tp / n_flipped
    out["f1"] = 0.0 if tp == 0 else 2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"])
    out["score_auc"] = float(roc_auc_score(is_flipped, score))
    top = np.argsort(-score)[:n_flipped]                           # the k most suspicious, k = number of flips
    out["precision_at_k"] = float(is_flipped[top].mean())
    out["k"] = n_flipped
    return out
