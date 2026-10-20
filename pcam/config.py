"""Load an experiment config from YAML into typed dataclasses.

Why dataclasses instead of a plain dict? A typo in the YAML (e.g. `learning_rate`
instead of `lr`) raises a clear error here, instead of silently falling back to a
default and producing a misleading run.
"""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

MODEL_NAMES = ("small_cnn", "resnet18")


class ConfigError(ValueError):
    """Raised when a config file is missing keys, has unknown keys, or bad values."""


@dataclass
class DataConfig:
    root: str = "data"              # where PCAM is downloaded / read from
    subset_fraction: float = 1.0    # fraction of train/val to use (test is always full)
    num_workers: int = 2            # DataLoader worker processes
    download: bool = True           # download PCAM if not already in `root`

    def __post_init__(self):
        if not 0 < self.subset_fraction <= 1:
            raise ConfigError(f"data.subset_fraction must be in (0, 1], got {self.subset_fraction}")
        if self.num_workers < 0:
            raise ConfigError(f"data.num_workers must be >= 0, got {self.num_workers}")


@dataclass
class NoiseConfig:
    flip_fraction: float = 0.0      # share of train labels to flip (0 = clean labels)
    seed: int = 0                   # picks which labels flip; independent of the training seed

    def __post_init__(self):
        if not 0 <= self.flip_fraction <= 0.5:
            raise ConfigError(f"noise.flip_fraction must be in [0, 0.5], got {self.flip_fraction}")


@dataclass
class ModelConfig:
    name: str = "small_cnn"         # one of MODEL_NAMES
    pretrained: bool = True         # only used by resnet18 (ImageNet weights)

    def __post_init__(self):
        if self.name not in MODEL_NAMES:
            raise ConfigError(f"Unknown model.name '{self.name}'; choose from: {', '.join(MODEL_NAMES)}")


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4

    def __post_init__(self):
        if self.epochs < 1:
            raise ConfigError(f"training.epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ConfigError(f"training.batch_size must be >= 1, got {self.batch_size}")
        if self.lr <= 0:
            raise ConfigError(f"training.lr must be > 0, got {self.lr}")


@dataclass
class Config:
    run_name: str
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _build(cls, values: dict, section: str):
    """Create dataclass `cls` from `values`, rejecting keys it doesn't define."""
    if not isinstance(values, dict):
        raise ConfigError(f"'{section}' must be a mapping, got {type(values).__name__}")
    allowed = {f.name for f in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in '{section}': {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )
    return cls(**values)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}

    if "run_name" not in raw:
        raise ConfigError("Config must define 'run_name'")

    sections = {
        "data": DataConfig,
        "noise": NoiseConfig,
        "model": ModelConfig,
        "training": TrainingConfig,
    }
    top = {k: v for k, v in raw.items() if k not in sections}
    nested = {name: _build(cls, raw.get(name, {}), name) for name, cls in sections.items()}
    return _build(Config, {**top, **nested}, "top level")
