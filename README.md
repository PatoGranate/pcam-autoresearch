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
| `noise.flip_fraction` | share of **train** labels to flip, 0 to 0.5 (default 0 = clean); val and test are never touched |
| `noise.seed` | picks which labels flip; independent of `seed`, so every model sees the same flips |
| `detection.folds` | `detect_noise.py` only: K for out-of-fold prediction (default 5) |
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

### Finding suspicious labels (Phase 2)

```bash
python detect_noise.py --config configs/detection/cnn_noise10.yaml
```

`detection.method` picks the detector:

- `out_of_fold` (default): trains `detection.folds` models; each predicts only the training
  patches it never saw. A label the models confidently disagree with is flagged (confident
  learning: per-class confidence thresholds).
- `training_loss` (`configs/detection/loss_noise*.yaml`): trains one model and records every
  patch's loss at every epoch. Patches whose loss stays high are suspicious; the mean
  P(given label) over epochs goes through the same confident-learning rule. About 5x cheaper.

With injected noise both score the flags against the known flips. Output folder:

| file | what it shows |
|---|---|
| `detection_metrics.json` | flags vs known flips: precision, recall, F1, AUC of the suspicion score, precision@k (k = number of flips); with clean labels only the number flagged |
| `suspicion_scores.csv` | per patch: index, given label, out-of-fold P(tumour), suspicion (1 − P(given label)), flagged, is_flipped |
| `flagged_indices.npy` | original train-split indices of the flagged patches |
| `loss_history.npy` | `training_loss` only: per-patch loss, shape (epochs, patches), same order as the CSV (`oof_prob_tumour` there is the epoch-averaged P(tumour) for this method) |

### Training on cleaned labels (Phase 2)

Any training config can drop or down-weight the patches a detector flagged:

```yaml
cleaning:
  mode: remove            # none (default) | remove | downweight
  flagged_path: runs/<detector run>/flagged_indices.npy
  downweight_factor: 0.1  # downweight only: loss weight of a flagged patch
```

Flags are original-split indices, so they only fit the subset they were made on (same
`seed` and `data.subset_fraction`); a mismatch stops with an error. `metrics.json` gets a
`cleaning` entry: patches flagged, how many of them were truly flipped, patches removed.

One command runs the whole comparison (about 35 min per seed with the default config):

```bash
python scripts/run_cleaning_matrix.py --seeds 42 43 44
```

Per seed it runs the training-loss detector, then trains four arms from
`configs/cleaning/base.yaml`: `clean` (no noise), `noisy`, `remove`, `downweight`. It is
resumable (`runs/cleaning_matrix/state.json`) and writes `runs/cleaning_matrix/summary.json`
with mean � sd of the test metrics per arm.

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
| `flipped_indices.npy` | only when noise is on: original train-split indices of the flipped labels (the ground truth for detectors); `metrics.json` also records the `noise` settings |

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

## Inspecting flagged patches (Phase 2e)

Run both detectors with `flip_fraction: 0` (`configs/detection/cnn_noise0.yaml`, `loss_noise0.yaml`),
then:

```
python scripts/inspect_flagged.py --oof-run runs/oof_noise0_<stamp> --loss-run runs/loss_noise0_<stamp>
```

It writes `runs/inspect_flagged_<stamp>/` with `suspects.csv` (fill in the `your_verdict` column),
`suspects_top.png` (patches both detectors flag, most suspicious first) and `suspects_random.png`
(random flagged patches). Judge each patch by the dashed central 32x32 box only.

## Phase 2 summary

Phase 2 asks: if some training labels are wrong, can we find them, and does removing them help?
The steps, each with its own section above:

1. **Inject noise** (`noise.flip_fraction`): flip a known share of train labels.
2. **Detect** (`detect_noise.py`): out-of-fold confident learning or training-loss ranking,
   scored against the known flips.
3. **Clean and compare** (`scripts/run_cleaning_matrix.py`): clean / noisy / remove / downweight.
4. **Inspect** (`scripts/inspect_flagged.py`): look at what the detectors flag on the real labels.

All numbers (noise effect, detector precision/recall, cleaning matrix, flag counts) are collected
from the run folders into one page, with no hand-written conclusions:

```bash
python scripts/phase2_results.py        # writes runs/phase2_results.html
```

Phase 2 experiments use `data.subset_fraction: 0.1` and a small CNN, so results show the method
working, not final accuracy. Everything is config in, `metrics.json` / `detection_metrics.json` out.

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
                inspect_flagged.py: grids + CSV of patches flagged on the real labels
                phase2_results.py: Phase 2 numbers -> runs/phase2_results.html
                run_cleaning_matrix.py: noisy vs cleaned training, several seeds
pcam/
  config.py     load + validate configs
  data.py       PCam datasets, subsets, augmentation, DataLoaders
  noise.py      flip a known fraction of train labels (Phase 2)
  noise_detection.py  out-of-fold + training-loss detectors, confident-learning flags, scoring vs known flips
  models.py     SmallCNN and ResNet18
  engine.py     training and evaluation loops
  metrics.py    accuracy, AUC, precision, recall, F1
  plots.py      curves, confusion matrix, worst-mistakes grid
  utils.py      seeding, device selection, run folders
train.py        one experiment, end to end
detect_noise.py find suspicious training labels (out-of-fold or training-loss)
```
