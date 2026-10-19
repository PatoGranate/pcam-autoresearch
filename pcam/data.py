"""PatchCamelyon data: datasets, subsetting, transforms and DataLoaders.

PCam patches are 96x96 RGB crops of lymph-node tissue. Label 1 means the
*central 32x32 region* contains tumour; tumour elsewhere in the patch doesn't count.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import PCAM

from pcam.config import Config
from pcam.utils import seed_worker

# ImageNet statistics: what the pretrained ResNet18 expects. We use them for the
# small CNN too, so swapping models never requires touching the data pipeline.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomRot90:
    """Rotate by a random multiple of 90 degrees.

    Tissue has no "up", so every rotation is an equally valid patch. Multiples of
    90 degrees are exact (no interpolation blur, no empty corners).
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        k = int(torch.randint(0, 4, (1,)))
        return torch.rot90(img, k, dims=(1, 2))


def build_transform(train: bool) -> transforms.Compose:
    steps = [transforms.ToTensor()]           # PIL image -> float tensor in [0, 1], shape (3, 96, 96)
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRot90(),
        ]
    steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(steps)


class PCamPatches(Dataset):
    """A (possibly subsetted) PCam split that returns (image, label, index).

    `index` is the patch's position in the *original* split, so it identifies the
    same patch regardless of which subset we drew. Phase 2 relies on this to track
    per-patch losses and which labels were flipped.
    """

    def __init__(self, base: PCAM, indices: np.ndarray, transform):
        self.base = base
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        index = int(self.indices[i])
        image, label = self.base[index]
        return self.transform(image), label, index


def sample_indices(n: int, fraction: float, seed: int) -> np.ndarray:
    """Pick a random `fraction` of range(n). Same seed -> same subset."""
    if fraction >= 1.0:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    k = max(1, int(round(n * fraction)))
    return np.sort(rng.choice(n, size=k, replace=False))  # sorted = friendlier disk reads


def build_datasets(cfg: Config) -> dict[str, PCamPatches]:
    datasets = {}
    for split in ("train", "val", "test"):
        base = PCAM(cfg.data.root, split=split, download=cfg.data.download)
        # Test always uses every patch, so test metrics are comparable across runs.
        fraction = 1.0 if split == "test" else cfg.data.subset_fraction
        indices = sample_indices(len(base), fraction, cfg.seed)
        datasets[split] = PCamPatches(base, indices, build_transform(train=(split == "train")))
    return datasets


def build_dataloaders(cfg: Config, device: torch.device) -> dict[str, DataLoader]:
    datasets = build_datasets(cfg)
    generator = torch.Generator().manual_seed(cfg.seed)  # makes the shuffle order reproducible
    workers = cfg.data.num_workers

    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=(device.type == "cuda"),   # faster CPU -> GPU copies
            persistent_workers=workers > 0,       # don't restart workers every epoch
            worker_init_fn=seed_worker,
            generator=generator,
        )
    return loaders
