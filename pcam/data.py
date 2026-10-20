"""PatchCamelyon data: datasets, subsetting, transforms and DataLoaders.

PCam patches are 96x96 RGB crops of lymph-node tissue. Label 1 means the
*central 32x32 region* contains tumour; tumour elsewhere in the patch doesn't count.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import PCAM

from pcam.config import Config
from pcam.noise import inject_label_noise
from pcam.utils import seed_worker

# ImageNet statistics: what the pretrained ResNet18 expects. We use them for the
# small CNN too, so swapping models never requires touching the data pipeline.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# File-name part for each split, and number of patches in each split
SPLIT_FILE_NAMES = {"train": "train", "val": "valid", "test": "test"}
SPLIT_SIZES = {"train": 262_144, "val": 32_768, "test": 32_768}


class RandomRot90:
    """Rotate by a random multiple of 90 degrees.

    Tissue has no "up", so every rotation is an equally valid patch. Multiples of
    90 degrees are exact (no interpolation blur, no empty corners).
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        k = int(torch.randint(0, 4, (1,)))
        return torch.rot90(img, k, dims=(1, 2))


def build_transform(train: bool) -> transforms.Compose:
    steps = [transforms.ToTensor()]           # uint8 (96, 96, 3) -> float tensor in [0, 1], shape (3, 96, 96)
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

    We read the HDF5 files ourselves rather than through torchvision's PCAM class:
    - PCAM can't be pickled, which breaks DataLoader workers on Windows.
    - PCAM reopens the file for every patch; we open it once per worker.
    - Labels live in memory as a numpy array (`self.labels`), so Phase 2 can
      inject label noise by editing that array.
    """

    def __init__(self, root: str, split: str, indices: np.ndarray, transform):
        name = SPLIT_FILE_NAMES[split]
        folder = Path(root) / "pcam"
        self.images_path = folder / f"camelyonpatch_level_2_split_{name}_x.h5"
        with h5py.File(folder / f"camelyonpatch_level_2_split_{name}_y.h5", "r") as f:
            self.labels = f["y"][:].reshape(-1).astype(np.int64)   # stored as (N, 1, 1, 1)
        self.indices = indices
        self.transform = transform
        self._images = None   # opened lazily, separately in each worker process

    def load_image(self, index: int) -> np.ndarray:
        """Raw patch as a (96, 96, 3) uint8 array, without any transform."""
        if self._images is None:
            self._images = h5py.File(self.images_path, "r")["x"]
        return self._images[index]

    def __getstate__(self):
        # An open HDF5 handle can't be sent to a worker process; each opens its own.
        state = self.__dict__.copy()
        state["_images"] = None
        return state

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        index = int(self.indices[i])
        return self.transform(self.load_image(index)), self.labels[index], index


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
        # torchvision handles the download (and skips it if the files exist)
        PCAM(cfg.data.root, split=split, download=cfg.data.download)
        n = SPLIT_SIZES[split]
        # Test always uses every patch, so test metrics are comparable across runs.
        fraction = 1.0 if split == "test" else cfg.data.subset_fraction
        indices = sample_indices(n, fraction, cfg.seed)
        datasets[split] = PCamPatches(cfg.data.root, split, indices,
                                      build_transform(train=(split == "train")))

    # Label noise goes into the train split only. `flipped_indices` are original-split
    # patch indices, the same ids the Dataset returns, so detectors can be scored against them.
    train = datasets["train"]
    train.clean_labels = train.labels.copy()
    train.flipped_indices = inject_label_noise(train.labels, train.indices,
                                               cfg.noise.flip_fraction, cfg.noise.seed)
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
