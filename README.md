# pcam-autoresearch

Tumour detection on [PatchCamelyon (PCam)](https://github.com/basveeling/pcam): 96×96 px
patches of lymph-node tissue, labelled **tumour** if the **central 32×32 region** contains
tumour tissue. Phase 1 (this code) is a config-driven baseline classifier with evaluation.
Later phases add label-noise detection and an LLM research agent.

## Install

Requires Python 3.10+. From the repo folder:

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate
```

Install PyTorch **first**, with the build that matches your machine:

```bash
# NVIDIA GPU (Windows / Linux)
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
# Apple Silicon (MPS) or CPU only
pip install torch==2.14.1 torchvision==0.29.1
```

Then the rest:

```bash
pip install -r requirements.txt
```

Check that the GPU is visible (`cuda` or `mps`, otherwise `cpu`):

```bash
python -c "from pcam.utils import get_device; print(get_device())"
```

## Run an experiment

```bash
python train.py --config configs/smoke.yaml      # ~1 min: checks everything works
python train.py --config configs/baseline.yaml   # small CNN, full data
python train.py --config configs/resnet18.yaml   # fine-tune pretrained ResNet18
```

**First run:** the dataset (~7.5 GB compressed, ~15 GB on disk with the archives) is
downloaded from Google Drive into `data/pcam/`. Later runs reuse it. If Google Drive refuses
with a quota error, download the `.h5.gz` files manually from the
[PCam repo](https://github.com/basveeling/pcam) into `data/pcam/` and run again.

To try something new, copy a config, change values, and run it. Every hyperparameter lives
in the YAML file:

| key | meaning |
|---|---|
| `run_name` | name prefix of the output folder |
| `seed` | seed for weights init, subset choice, shuffling and augmentation |
| `data.subset_fraction` | fraction of train/val to use (e.g. `0.1`); test always uses all 32,768 patches |
| `data.num_workers` | parallel data-loading processes (`0` = load in the main process) |
| `model.name` | `small_cnn` or `resnet18` |
| `model.pretrained` | `resnet18` only: start from ImageNet weights |
| `training.epochs`, `batch_size`, `lr`, `weight_decay` | optimiser settings (AdamW) |

Unknown keys or invalid values stop the run immediately with a clear error message.

### Seed sweep and summary page

`configs/seed_sweep/` holds the CNN and ResNet18 configs for seeds 43 and 44 (seed 42 is
`baseline.yaml` / `resnet18.yaml`). Run them **one at a time** (each is 25+ min on a laptop GPU,
and two at once doubles memory use):

```bash
python train.py --config configs/seed_sweep/baseline_cnn_s43.yaml
python train.py --config configs/seed_sweep/baseline_cnn_s44.yaml
python train.py --config configs/seed_sweep/resnet18_pretrained_s43.yaml
python train.py --config configs/seed_sweep/resnet18_pretrained_s44.yaml
```

Then summarise every finished full-data run (subset runs are skipped) into one page:

```bash
python scripts/summarize_runs.py        # writes runs/report.html
```

It shows the loss curves per seed, a metrics table with mean ± sd per model, and each
model's worst mistakes.

## Reading the outputs

Each run writes to `runs/<run_name>_<timestamp>/`:

| file | what it shows |
|---|---|
| `config.yaml` | the exact settings used, including defaults |
| `metrics.json` | test accuracy, **AUC**, precision, recall, F1, confusion counts; best epoch; run time |
| `curves.png` | train vs validation loss and accuracy per epoch |
| `history.json` | the numbers behind `curves.png` (plus val AUC per epoch) |
| `confusion_matrix.png` | test predictions: rows = true class, columns = predicted |
| `worst_mistakes.png` | the 16 test patches the model got wrong most confidently |
| `test_predictions.csv` | every test patch: index, true label, predicted P(tumour) |
| `best_model.pt` | weights from the epoch with the best validation AUC |

How to read them:

- **AUC is the main metric.** It measures how well the model ranks tumour above normal
  patches, independent of the 0.5 threshold. 0.5 = random, 1.0 = perfect.
- **Recall** (sensitivity) is the fraction of tumour patches caught; in a screening setting a
  missed tumour (false negative) is usually worse than a false alarm.
- **Curves:** validation loss rising while training loss keeps falling means overfitting.
  The test metrics come from the best-validation-AUC epoch, not the last one.
- **Worst mistakes:** the dashed yellow box is the region the label describes. Check whether
  the tumour (or its absence) is actually inside the box. Some confident "mistakes" may be
  labelling errors rather than model errors.

## Reproducibility

- Seeds are fixed for Python, NumPy and PyTorch, including DataLoader workers; the same
  config picks the same subset and shuffle order.
- On GPU, results are very close but not always bit-identical, because some CUDA kernels
  are non-deterministic.
- `requirements.txt` pins exact package versions.

## Project layout

```
configs/        experiment configs (YAML); seed_sweep/ has the extra seeds
scripts/        summarize_runs.py: finished runs -> runs/report.html
pcam/
  config.py     load + validate configs
  data.py       PCam datasets, subsets, augmentation, DataLoaders
  models.py     SmallCNN and ResNet18
  engine.py     training and evaluation loops
  metrics.py    accuracy, AUC, precision, recall, F1
  plots.py      curves, confusion matrix, worst-mistakes grid
  utils.py      seeding, device selection, run folders
train.py        one experiment, end to end
```
