# CLAUDE.md

## Project: PCam label-noise detection + research agent

The owner is an MSc AI student (Python background, new to Claude Code). The project has three
phases. **Only build what the current phase needs**, but keep the code structured so later
phases fit in cleanly.

**Current phase: Phase 1.**

### Phase 1 — Baseline classifier
Binary tumour classifier on PatchCamelyon (`torchvision.datasets.PCAM`).

- Data: train/val/test splits, with an option to use a small random subset (e.g. 10%) for fast
  development runs.
- Model: a small CNN, plus an option to fine-tune a pretrained ResNet18.
- Config-driven: all hyperparameters (model, lr, batch size, epochs, subset fraction, seed) live
  in a YAML file. One command per run: `python train.py --config configs/baseline.yaml`.
- Each run saves into `runs/<run_name>/`: the config used, train/val loss and accuracy curves
  (PNG), test metrics (accuracy, AUC, precision, recall, F1) in `metrics.json`, a confusion
  matrix, and a grid of the most confidently wrong test patches.
- Reproducibility: fixed seeds, `requirements.txt`, and a short README (install, run, read
  outputs).
- Device: use CUDA or Apple MPS if available, otherwise CPU.

### Phase 2 — Label-noise detection
Find mislabelled training patches and measure whether removing or down-weighting them improves
the model.

- Inject synthetic label noise: flip a known fraction (e.g. 5–20%) of training labels and record
  which ones were flipped.
- Detect suspicious labels, e.g. via out-of-fold predicted probabilities (confident learning),
  per-sample loss during training, or disagreement between models.
- Evaluate the detector against the known flips (precision/recall of detected noise).
- Compare training on noisy vs cleaned data (test AUC/accuracy).
- Inspect real (non-injected) flagged patches visually. PCam's labelling rule: a patch is
  positive if its **central 32x32 region** contains tumour.
- Keep the clean interface: every experiment = config in, `metrics.json` out.

### Phase 3 — The owner's own LLM research agent
A small Python agent (~200–400 lines) using the Anthropic API: a message loop plus a few tools
(`read_file`, `write_file`, `list_files`, `run_command` restricted to this repo, with
confirmation before running commands). It will autonomously run Phase 2 experiments, read
`metrics.json`, change configs/code, iterate toward better test AUC or noise detection, and keep
notes in `notes.md`.

**Phase 3 is a learning exercise. Do not write the agent.** Act as a tutor: explain concepts,
review the owner's code, and suggest fixes, but let them write the core loop and tool-calling
logic themselves.

## Working style

- Explain key decisions briefly; the owner wants to understand the code, not just run it.
- Keep code readable over clever.
- Propose a plan and wait for an OK before large changes.
- Keep changes small and testable; run a quick smoke test (tiny subset, 1 epoch) after building
  something.

## Git workflow

Claude works as the GitHub account `pelayo-claude-bot`. The repo owner reviews and merges every change.

- **Never commit to `main` directly.** Start each task from an up-to-date `main` on a new branch named `claude/<short-description>` (e.g. `claude/add-guardrails`).
- Commit on that branch, push it to `origin`, and open a pull request against `main` with `gh pr create`.
- **Do not merge, approve, or close PRs.** The owner reviews and merges. If changes are requested, push follow-up commits to the same branch rather than opening a new PR.
- Never force-push, rewrite history on pushed branches, or delete branches you didn't create.
- One logical change per PR. Keep the PR description short: what changed, why, and how it was verified.

## Guardrails

`.claude/settings.json` enforces this workflow: pushing to `main`, force-pushing, merging or approving PRs, deleting the repo, and raw `gh api` calls are denied; `git push` and `gh pr create` require confirmation. Don't try to work around these rules. If one blocks something that seems necessary, stop and ask.
