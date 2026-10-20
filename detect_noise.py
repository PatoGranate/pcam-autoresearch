"""Find suspicious training labels: python detect_noise.py --config configs/detection/cnn_noise10.yaml

Trains K fold models on the (noisy) training subset, predicts each patch with the
model that never saw it, flags labels the models confidently disagree with, and
scores the flags against the known flips when label noise was injected.
Writes everything into runs/<run_name>_<timestamp>/ ; the last line printed is that folder.
"""

import argparse
import json
import sys
import time

import numpy as np
import yaml

from pcam.config import ConfigError, load_config
from pcam.data import build_datasets
from pcam.noise_detection import (confident_learning_flags, out_of_fold_probs,
                                  score_detector, suspicion_score)
from pcam.utils import get_device, make_run_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect suspicious training labels (out-of-fold).")
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(2)

    start = time.time()
    set_seed(cfg.seed)
    device = get_device()
    run_dir = make_run_dir(cfg.run_name)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    print(f"Run folder: {run_dir}\nDevice: {device}")

    train_ds = build_datasets(cfg)["train"]          # noise (if configured) is already applied
    indices = train_ds.indices
    given = train_ds.labels[indices]                 # labels the detector sees
    is_flipped = np.isin(indices, train_ds.flipped_indices)   # ground truth (all False without noise)
    print(f"Train patches: {len(indices):,} | injected flips: {int(is_flipped.sum()):,}")

    probs, folds = out_of_fold_probs(train_ds, cfg, device)

    score = suspicion_score(given, probs)
    flagged = confident_learning_flags(given, probs)
    result = score_detector(is_flipped, score, flagged)

    # One row per patch: everything a later step (cleaning, visual inspection) needs.
    np.savetxt(run_dir / "suspicion_scores.csv",
               np.column_stack([indices, given, probs, score, flagged, is_flipped]),
               delimiter=",", header="index,given_label,oof_prob_tumour,suspicion,flagged,is_flipped",
               comments="", fmt=["%d", "%d", "%.6f", "%.6f", "%d", "%d"])
    np.save(run_dir / "flagged_indices.npy", indices[flagged])

    metrics = {
        "run_name": cfg.run_name,
        "run_dir": str(run_dir),
        "model": cfg.model.name,
        "device": str(device),
        "noise": {"flip_fraction": cfg.noise.flip_fraction, "seed": cfg.noise.seed},
        "folds": cfg.detection.folds,
        "epochs_per_fold": cfg.training.epochs,
        "fold_summaries": folds,
        "detection": result,
        "total_seconds": round(time.time() - start, 1),
    }
    (run_dir / "detection_metrics.json").write_text(json.dumps(metrics, indent=2))

    if result["n_flipped"]:
        print(f"Detection: flagged {result['n_flagged']:,} | precision {result['precision']:.3f} | "
              f"recall {result['recall']:.3f} | f1 {result['f1']:.3f} | "
              f"score AUC {result['score_auc']:.3f} | precision@{result['k']} {result['precision_at_k']:.3f}")
    else:
        print(f"Detection (clean labels): flagged {result['n_flagged']:,} "
              f"({result['flagged_fraction']:.1%} of train patches)")
    print(run_dir)


if __name__ == "__main__":   # required on Windows when DataLoader uses worker processes
    main()
