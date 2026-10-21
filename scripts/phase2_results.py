"""Phase 2 results page: python scripts/phase2_results.py   (writes runs/phase2_results.html)

Collects the numbers of every Phase 2 step from the run folders, with no hand-written conclusions:
  1. noise injection:  test metrics of the CNN trained on 0 / 10 / 20% flipped labels
  2. detectors:        out-of-fold and training-loss detectors vs the known flips
  3. cleaning:         clean / noisy / remove / downweight, mean +- sd over seeds
  4. inspection:       how many real (un-injected) labels the detectors flag
For each run name the newest folder runs/<name>_<timestamp>/ is used. Missing pieces are skipped.
"""

import argparse
import csv
import html
import json
from pathlib import Path

NOISE_LEVELS = (0, 10, 20)
ARMS = ("clean", "noisy", "remove", "downweight")


def newest(runs_dir: Path, name: str, filename: str) -> Path | None:
    """Newest runs/<name>_<timestamp>/<filename>; the timestamp suffix keeps names from clashing."""
    found = sorted(p for p in runs_dir.glob(f"{name}_2*/{filename}") if p.parent.name[len(name) + 1:][:1].isdigit())
    return found[-1] if found else None


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def noise_section(runs_dir: Path) -> str:
    rows = []
    for pct in NOISE_LEVELS:
        path = newest(runs_dir, f"cnn_noise{pct}", "metrics.json")
        if path:
            t = read_json(path)["test"]
            rows.append([f"{pct}%"] + [f"{t[k]:.3f}" for k in ("auc", "accuracy", "precision", "recall", "f1")])
    if not rows:
        return ""
    return ("<h2>1. Effect of label noise</h2><p>Small CNN, 10% subset, seed 42; labels flipped in train only.</p>"
            + table(["flipped", "test AUC", "accuracy", "precision", "recall", "F1"], rows))


def detector_section(runs_dir: Path) -> str:
    rows, clean = [], []
    for method, prefix in (("out-of-fold", "oof"), ("training loss", "loss")):
        for pct in NOISE_LEVELS:
            path = newest(runs_dir, f"{prefix}_noise{pct}", "detection_metrics.json")
            if not path:
                continue
            d = read_json(path)["detection"]
            if pct == 0:
                clean.append([method, f"{d['n_flagged']:,}", f"{d['flagged_fraction']:.1%}"])
            else:
                rows.append([method, f"{pct}%", f"{d['n_flagged']:,}", f"{d['precision']:.3f}", f"{d['recall']:.3f}",
                             f"{d['f1']:.3f}", f"{d['score_auc']:.3f}", f"{d['precision_at_k']:.3f}"])
    out = ""
    if rows:
        out += ("<h2>2. Detectors vs the known flips</h2>"
                + table(["method", "flipped", "flagged", "precision", "recall", "F1", "score AUC", "precision@k"], rows))
    if clean:
        out += ("<h3>Clean labels (nothing injected): every flag is a false alarm or a real suspect</h3>"
                + table(["method", "flagged", "share of train"], clean))
    return out


def cleaning_section(runs_dir: Path) -> str:
    path = runs_dir / "cleaning_matrix" / "summary.json"
    if not path.exists():
        return ""
    summary = read_json(path)
    rows = []
    for arm in ARMS:
        if arm in summary:
            s = summary[arm]
            rows.append([arm, str(s["n_seeds"])] + [f"{s[k]['mean']:.3f} &plusmn; {s[k]['sd']:.3f}"
                                                     for k in ("auc", "accuracy", "f1")])
    return ("<h2>3. Noisy vs cleaned training</h2><p>10% flipped labels, 10% subset; mean &plusmn; sd over seeds. "
            "With so few seeds the sd is as large as the gaps between arms, so read differences carefully.</p>"
            + table(["arm", "seeds", "test AUC", "accuracy", "F1"], rows))


def inspection_section(runs_dir: Path) -> str:
    found = sorted(runs_dir.glob("inspect_flagged_2*/suspects.csv"))
    if not found:
        return ""
    with open(found[-1], encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    both = sum(r["oof_flagged"] == "True" and r["loss_flagged"] == "True" for r in rows)
    tumour = sum(r["given_label"] == "1" for r in rows)
    verdicts = [r["your_verdict"].strip() for r in rows if r.get("your_verdict", "").strip()]
    text = (f"<p>{len(rows):,} real patches flagged by either detector, {both:,} by both; "
            f"{tumour:,} are labelled tumour, {len(rows) - tumour:,} normal. Folder: <code>{found[-1].parent.name}</code>.</p>")
    if verdicts:
        counts = {v: verdicts.count(v) for v in sorted(set(verdicts))}
        text += (f"<p>Manual verdicts ({len(verdicts)} rows filled in): "
                 + ", ".join(f"{html.escape(v)}: {n}" for v, n in counts.items()) + ".</p>")
    else:
        text += "<p>No manual verdicts recorded yet (<code>your_verdict</code> column is empty).</p>"
    return "<h2>4. Real flagged patches</h2>" + text


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Phase 2 results</title>
<style>
body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 52rem; margin: 2rem auto; padding: 0 1rem; }}
table {{ border-collapse: collapse; margin: 0.5rem 0 1.5rem; }}
th, td {{ border: 1px solid #8884; padding: 0.3rem 0.7rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
thead th {{ background: #8882; }}
</style></head><body>
<h1>Phase 2: label noise</h1>
{sections}
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Phase 2 results into one HTML page.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default=None, help="output file (default: <runs-dir>/phase2_results.html)")
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)

    sections = "".join(f(runs_dir) for f in (noise_section, detector_section, cleaning_section, inspection_section))
    if not sections:
        raise SystemExit(f"No Phase 2 results found in {runs_dir}/")
    out = Path(args.out) if args.out else runs_dir / "phase2_results.html"
    out.write_text(PAGE.format(sections=sections), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
