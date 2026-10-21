"""Noisy vs cleaned training: python scripts/run_cleaning_matrix.py --seeds 42 43 44

For each seed, runs one detector pass on the noisy labels and then trains four arms:
  clean       no label noise (the ceiling)
  noisy       10% flipped labels, nothing done about it (the floor)
  remove      noisy labels, flagged patches dropped
  downweight  noisy labels, flagged patches count for less in the loss
Everything runs one process at a time. Progress is kept in runs/cleaning_matrix/state.json,
so an interrupted run resumes where it stopped. Prints mean +- sd of test metrics per arm
and writes runs/cleaning_matrix/summary.json.
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "cleaning_matrix"
ARMS = ("clean", "noisy", "remove", "downweight")
METRICS = ("auc", "accuracy", "precision", "recall", "f1")


def run(script: str, config: dict, name: str) -> Path:
    """Write `config` to a YAML file, run `python <script> --config <it>`, return the run folder."""
    (OUT / "configs").mkdir(parents=True, exist_ok=True)
    path = OUT / "configs" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    proc = subprocess.run([sys.executable, script, "--config", str(path)], cwd=ROOT,
                          stdout=subprocess.PIPE, text=True)   # stderr goes straight to the terminal
    log = OUT / "logs"
    log.mkdir(exist_ok=True)
    (log / f"{name}.log").write_text(proc.stdout)
    if proc.returncode != 0:
        sys.exit(f"{script} failed for {name} (see {log / (name + '.log')})")
    return ROOT / proc.stdout.strip().splitlines()[-1]   # last line printed = run folder


def summarize(state: dict, seeds: list[int]) -> dict:
    summary = {}
    for arm in ARMS:
        rows = []
        for seed in seeds:
            folder = state.get(f"{seed}/{arm}")
            if folder:
                rows.append(json.loads((ROOT / folder / "metrics.json").read_text())["test"])
        if rows:
            summary[arm] = {"n_seeds": len(rows)}
            for key in METRICS:
                vals = [r[key] for r in rows]
                summary[arm][key] = {"mean": statistics.mean(vals),
                                     "sd": statistics.stdev(vals) if len(vals) > 1 else None,
                                     "per_seed": vals}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the noisy-vs-cleaned training matrix.")
    parser.add_argument("--base", default="configs/cleaning/base.yaml")
    parser.add_argument("--out", default="runs/cleaning_matrix", help="folder for state, configs, logs, summary")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    global OUT
    OUT = ROOT / args.out

    base = yaml.safe_load((ROOT / args.base).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    state_path = OUT / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    def record(key: str, folder: Path) -> None:
        state[key] = str(folder.relative_to(ROOT))
        state_path.write_text(json.dumps(state, indent=2))

    for seed in args.seeds:
        def config(arm: str, **overrides) -> dict:
            cfg = json.loads(json.dumps(base))   # cheap deep copy
            cfg["seed"], cfg["run_name"] = seed, f"cleaning_{arm}_s{seed}"
            for section, values in overrides.items():
                cfg.setdefault(section, {}).update(values)
            return cfg

        if f"{seed}/detector" not in state:
            print(f"\n=== seed {seed}: detector ===", flush=True)
            record(f"{seed}/detector", run("detect_noise.py", config("detector"), f"detector_s{seed}"))
        flagged = str(ROOT / state[f"{seed}/detector"] / "flagged_indices.npy")

        arm_overrides = {"clean": {"noise": {"flip_fraction": 0.0}},
                         "noisy": {},
                         "remove": {"cleaning": {"mode": "remove", "flagged_path": flagged}},
                         "downweight": {"cleaning": {"mode": "downweight", "flagged_path": flagged}}}
        for arm in ARMS:
            if f"{seed}/{arm}" in state:
                continue
            print(f"\n=== seed {seed}: {arm} ===", flush=True)
            record(f"{seed}/{arm}", run("train.py", config(arm, **arm_overrides[arm]), f"{arm}_s{seed}"))

    summary = summarize(state, args.seeds)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nTest metrics, mean +- sd over seeds {args.seeds}")
    for arm, s in summary.items():
        cells = " | ".join(f"{k} {s[k]['mean']:.4f}" + (f" +- {s[k]['sd']:.4f}" if s[k]["sd"] is not None else "")
                           for k in METRICS)
        print(f"{arm:<11} (n={s['n_seeds']}) {cells}")


if __name__ == "__main__":
    main()
