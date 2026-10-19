# CLAUDE.md

## Workflow

Claude works as the GitHub account `pelayo-claude-bot`. The repo owner reviews and merges every change.

- **Never commit to `main` directly.** Start each task from an up-to-date `main` on a new branch named `claude/<short-description>` (e.g. `claude/add-guardrails`).
- Commit on that branch, push it to `origin`, and open a pull request against `main` with `gh pr create`.
- **Do not merge, approve, or close PRs.** The owner reviews and merges. If changes are requested, push follow-up commits to the same branch rather than opening a new PR.
- Never force-push, rewrite history on pushed branches, or delete branches you didn't create.
- One logical change per PR. Keep the PR description short: what changed, why, and how it was verified.

## Guardrails

`.claude/settings.json` enforces this workflow: pushing to `main`, force-pushing, merging or approving PRs, deleting the repo, and raw `gh api` calls are denied; `git push` and `gh pr create` require confirmation. Don't try to work around these rules. If one blocks something that seems necessary, stop and ask.
