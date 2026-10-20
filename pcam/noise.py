"""Synthetic label noise: flip a known fraction of training labels.

Phase 2 needs ground truth for "which labels are wrong", so we create the wrong
labels ourselves and remember exactly which ones they are. Only the training
labels are ever touched; val and test stay clean so metrics remain comparable.
"""

import numpy as np


def inject_label_noise(labels: np.ndarray, indices: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Flip `fraction` of the labels of the patches in `indices` (in place).

    labels:   label array for the whole split (edited in place; binary 0/1)
    indices:  original-split positions of the patches actually used for training
    fraction: share of those patches to flip (0 flips nothing)
    seed:     same seed -> same flips, whichever model or training seed is used

    Returns the sorted original-split indices of the flipped patches.
    """
    n_flip = int(round(len(indices) * fraction))
    if n_flip == 0:
        return np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    flipped = np.sort(rng.choice(indices, size=n_flip, replace=False))
    labels[flipped] = 1 - labels[flipped]
    return flipped
