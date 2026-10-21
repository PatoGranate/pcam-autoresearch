"""Look at flagged patches of the real (un-injected) labels:

    python scripts/inspect_flagged.py --oof-run runs/oof_noise0_<stamp> --loss-run runs/loss_noise0_<stamp>

Takes finished detector runs made with flip_fraction 0 (so every flag is a *real* suspect, not an
injected flip) and writes into runs/inspect_flagged_<timestamp>/:
  suspects.csv         every patch flagged by either detector, most suspicious first
  suspects_top.png     grid of the top suspects, dashed box = the central 32x32 the label describes
  suspects_random.png  grid of random flagged patches (the top of the list can be unrepresentative)
No training happens here; it only reads the detectors' suspicion_scores.csv files.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so `pcam` imports from scripts/

from pcam.data import PCamPatches   # noqa: E402
from pcam.utils import make_run_dir  # noqa: E402

CLASS_NAMES = ("normal", "tumour")


def read_scores(run_dir: Path) -> dict[int, dict]:
    """suspicion_scores.csv -> {patch index: row}."""
    with open(run_dir / "suspicion_scores.csv", encoding="utf-8-sig") as f:
        return {int(r["index"]): r for r in csv.DictReader(f)}


def plot_grid(dataset: PCamPatches, rows: list[dict], title: str, path: Path, cols: int = 6) -> None:
    n_rows = max(1, int(np.ceil(len(rows) / cols)))
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 2.3, n_rows * 2.7))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, row in zip(np.ravel(axes), rows):
        ax.imshow(dataset.load_image(row["index"]))
        ax.add_patch(Rectangle((32, 32), 32, 32, fill=False, edgecolor="yellow",
                               linestyle="--", linewidth=1.2))
        both = "both" if row["oof_flagged"] and row["loss_flagged"] else ("oof" if row["oof_flagged"] else "loss")
        ax.set_title(f"#{row['index']} label: {CLASS_NAMES[row['given_label']]}\n"
                     f"P(tumour) {row['oof_prob']:.2f} | {both}", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect flagged patches of the real labels.")
    parser.add_argument("--oof-run", required=True, type=Path, help="out-of-fold detector run (noise 0)")
    parser.add_argument("--loss-run", required=True, type=Path, help="training-loss detector run (noise 0)")
    parser.add_argument("--n", type=int, default=24, help="patches per grid")
    parser.add_argument("--seed", type=int, default=0, help="seed for the random grid")
    args = parser.parse_args()

    oof, loss = read_scores(args.oof_run), read_scores(args.loss_run)
    shared = sorted(oof.keys() & loss.keys())
    if len(shared) != len(oof) or len(shared) != len(loss):
        print(f"Warning: runs cover different patches ({len(oof)} vs {len(loss)}); using the {len(shared)} shared")

    rows = []
    for i in shared:
        o, l = oof[i], loss[i]
        if o["flagged"] == "0" and l["flagged"] == "0":
            continue
        rows.append({
            "index": i,
            "given_label": int(o["given_label"]),
            "oof_prob": float(o["oof_prob_tumour"]),
            "oof_suspicion": float(o["suspicion"]),
            "oof_flagged": o["flagged"] == "1",
            "loss_prob": float(l["oof_prob_tumour"]),
            "loss_suspicion": float(l["suspicion"]),
            "loss_flagged": l["flagged"] == "1",
        })
    # Both detectors agreeing is the strongest evidence, so those come first; then by average suspicion.
    rows.sort(key=lambda r: (-(r["oof_flagged"] and r["loss_flagged"]), -(r["oof_suspicion"] + r["loss_suspicion"])))

    out = make_run_dir("inspect_flagged")
    with open(out / "suspects.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) + ["your_verdict"])   # owner fills in the last column
        writer.writeheader()
        writer.writerows(rows)

    train = PCamPatches("data", "train", np.array([], dtype=int), transform=None)
    n_both = sum(r["oof_flagged"] and r["loss_flagged"] for r in rows)
    print(f"Flagged by either: {len(rows):,} of {len(shared):,} | by both: {n_both:,} "
          f"| given label tumour: {sum(r['given_label'] for r in rows):,}")
    top = rows[: args.n]
    plot_grid(train, top, f"Top {len(top)} suspects (both detectors first); dashed = labelled region",
              out / "suspects_top.png")
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(rows), size=min(args.n, len(rows)), replace=False)
    plot_grid(train, [rows[i] for i in pick], f"{len(pick)} random flagged patches; dashed = labelled region",
              out / "suspects_random.png")
    print(out)


if __name__ == "__main__":
    main()
