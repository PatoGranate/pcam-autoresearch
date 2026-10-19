"""Run one experiment: python train.py --config configs/baseline.yaml

Trains a model, keeps the checkpoint with the best validation AUC, evaluates it
on the test set, and writes everything into runs/<run_name>_<timestamp>/.
The last line printed is the run folder path.
"""

import argparse
import json
import sys
import time

import numpy as np
import torch
import yaml
from torch import nn

from pcam.config import ConfigError, load_config
from pcam.data import build_dataloaders
from pcam.engine import evaluate, train_one_epoch
from pcam.metrics import compute_metrics
from pcam.models import build_model, count_parameters
from pcam.plots import plot_confusion_matrix, plot_curves, plot_worst_mistakes
from pcam.utils import get_device, make_run_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a PCam classifier.")
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    return parser.parse_args()


def main() -> None:
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

    # --- Data and model -----------------------------------------------------
    loaders = build_dataloaders(cfg, device)
    sizes = {split: len(loader.dataset) for split, loader in loaders.items()}
    print(f"Patches: train {sizes['train']:,} | val {sizes['val']:,} | test {sizes['test']:,}")

    model = build_model(cfg.model).to(device)
    print(f"Model: {cfg.model.name} ({count_parameters(model):,} trainable parameters)")

    # BCEWithLogitsLoss = sigmoid + binary cross-entropy in one numerically stable step
    loss_fn = nn.BCEWithLogitsLoss()
    # AdamW: Adam with weight decay applied correctly (decoupled from the gradient)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                                  weight_decay=cfg.training.weight_decay)

    # --- Training -----------------------------------------------------------
    # Model selection uses validation AUC; the test set is only touched once, at the end.
    history = []
    best_auc, best_epoch = -1.0, 0
    checkpoint_path = run_dir / "best_model.pt"

    for epoch in range(1, cfg.training.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, loss_fn, device)
        val = evaluate(model, loaders["val"], loss_fn, device)
        val_metrics = compute_metrics(val.labels, val.probs)
        val_auc = val_metrics["auc"] if val_metrics["auc"] is not None else 0.0

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val.loss,
            "val_acc": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "seconds": round(time.time() - epoch_start, 1),
        })
        improved = val_auc > best_auc
        if improved:
            best_auc, best_epoch = val_auc, epoch
            torch.save(model.state_dict(), checkpoint_path)
        print(f"Epoch {epoch}/{cfg.training.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val.loss:.4f} acc {val_metrics['accuracy']:.4f} auc {val_auc:.4f} | "
              f"{history[-1]['seconds']:.0f}s{'  *best*' if improved else ''}")

    train_seconds = time.time() - start

    # --- Test evaluation with the best checkpoint ---------------------------
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test = evaluate(model, loaders["test"], loss_fn, device)
    test_metrics = compute_metrics(test.labels, test.probs)
    test_metrics["loss"] = test.loss

    # --- Save outputs -------------------------------------------------------
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    np.savetxt(run_dir / "test_predictions.csv",
               np.column_stack([test.indices, test.labels, test.probs]),
               delimiter=",", header="index,label,prob_tumour", comments="",
               fmt=["%d", "%d", "%.6f"])

    plot_curves(history, run_dir / "curves.png")
    plot_confusion_matrix(test_metrics["confusion_matrix"], run_dir / "confusion_matrix.png")
    plot_worst_mistakes(loaders["test"].dataset, test, run_dir / "worst_mistakes.png")

    metrics = {
        "run_name": cfg.run_name,
        "run_dir": str(run_dir),
        "model": cfg.model.name,
        "device": str(device),
        "n_train": sizes["train"],
        "n_val": sizes["val"],
        "n_test": sizes["test"],
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "test": test_metrics,
        "train_seconds": round(train_seconds, 1),
        "total_seconds": round(time.time() - start, 1),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Test (best epoch {best_epoch}): "
          f"acc {test_metrics['accuracy']:.4f} | auc {test_metrics['auc']:.4f} | "
          f"precision {test_metrics['precision']:.4f} | recall {test_metrics['recall']:.4f} | "
          f"f1 {test_metrics['f1']:.4f}")
    print(run_dir)   # last line: where the outputs are (handy for scripts and agents)


if __name__ == "__main__":   # required on Windows when DataLoader uses worker processes
    main()
