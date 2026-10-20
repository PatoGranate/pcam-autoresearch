"""Summarise finished runs into one HTML page: python scripts/summarize_runs.py

Finds every runs/*/metrics.json, keeps the full-data runs (subset_fraction 1.0, so
smoke tests and quick subset runs don't mix in), groups them by model and seed,
and writes runs/report.html: training curves, a per-seed metrics table with
mean +- sd, and each model's most confidently wrong test patches.

Needs Pillow (installed together with torchvision).
"""

import argparse
import base64
import io
import json
import statistics
import sys
from pathlib import Path

import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
METRIC_KEYS = ("auc", "accuracy", "precision", "recall", "f1")


def load_runs(runs_dir: Path) -> list[dict]:
    """One dict per finished full-data run: model, seed, metrics, history, folder."""
    runs = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        folder = metrics_path.parent
        cfg = yaml.safe_load((folder / "config.yaml").read_text())
        if cfg["data"]["subset_fraction"] < 1.0:
            continue
        runs.append({
            "folder": folder,
            "model": cfg["model"]["name"],
            "seed": cfg["seed"],
            "metrics": json.loads(metrics_path.read_text()),
            "history": json.loads((folder / "history.json").read_text()),
        })
    return runs


def jpeg_data_uri(path: Path, width: int = 900) -> str:
    """Shrink a PNG into an embedded JPEG so the page stays one small file."""
    img = Image.open(path).convert("RGB")
    img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def table_rows(runs: list[dict]) -> str:
    rows = []
    for model in sorted({r["model"] for r in runs}):
        mine = sorted((r for r in runs if r["model"] == model), key=lambda r: r["seed"])
        for r in mine:
            t = r["metrics"]["test"]
            cells = "".join(f"<td>{t[k]:.3f}</td>" for k in METRIC_KEYS)
            rows.append(f"<tr><th scope='row'>{model}</th><td>{r['seed']}</td>{cells}"
                        f"<td>{r['metrics']['best_epoch']}</td></tr>")
        if len(mine) > 1:   # sd needs at least two seeds
            cells = ""
            for k in METRIC_KEYS:
                values = [r["metrics"]["test"][k] for r in mine]
                cells += f"<td>{statistics.mean(values):.3f} &plusmn; {statistics.stdev(values):.3f}</td>"
            rows.append(f"<tr class='mean'><th scope='row'>{model}</th><td>mean &plusmn; sd</td>{cells}<td></td></tr>")
    return "".join(rows)


def build_payload(runs: list[dict]) -> dict:
    models = sorted({r["model"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})
    run_data = {}
    figures = {}
    for r in runs:
        run_data[f"{r['model']}|{r['seed']}"] = {
            "h": [{"e": h["epoch"], "tl": h["train_loss"], "vl": h["val_loss"], "va": h["val_auc"]}
                  for h in r["history"]],
            "t": r["metrics"]["test"],
        }
    for model in models:   # figures come from each model's lowest seed
        first = min((r for r in runs if r["model"] == model), key=lambda r: r["seed"])
        figures[model] = {
            "seed": first["seed"],
            "confusion": first["metrics"]["test"]["confusion_matrix"],
            "img": jpeg_data_uri(first["folder"] / "worst_mistakes.png"),
        }
    return {"models": models, "seeds": seeds, "runs": run_data, "figures": figures}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise finished runs into an HTML page.")
    parser.add_argument("--runs-dir", default="runs", help="folder that holds the run folders")
    parser.add_argument("--out", default=None, help="output file (default: <runs-dir>/report.html)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    runs = load_runs(runs_dir)
    if not runs:
        sys.exit(f"No finished full-data runs found in {runs_dir}/")

    html = (HERE / "report_template.html").read_text(encoding="utf-8")
    html = html.replace("/*DATA*/null", json.dumps(build_payload(runs)))
    html = html.replace("%%ROWS%%", table_rows(runs))
    out = Path(args.out) if args.out else runs_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"{len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
