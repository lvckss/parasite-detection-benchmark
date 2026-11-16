---
name: semantic-commits-and-push
description: Group existing changes into semantic commits on a feature branch, push it, and open a pull request when asked to commit and publish the work.
---

# Semantic commits, push, and pull request

Package existing work into coherent commits and a pull request for the team to review and merge. Use the user's invocation context to determine the intended scope, branch name, and optional commit-message context. Follow any narrower request, such as committing locally without pushing.

## Inspect the work

- Confirm the Git root, current branch, remotes, upstream, and recent commit style.
- Inspect `git status --short`, the staged and unstaged diffs, and relevant untracked files. Check notebook outputs as well as code cells before including a notebook.
- If there is no work to commit, report that and stop without pushing or opening a PR.
- Stop for unresolved conflicts, an unfinished Git operation, or detached HEAD. Preserve existing work and partial staging; clarify only when scope or ownership is ambiguous.
- Keep credentials, downloaded datasets, model weights, and bulky generated experiment artifacts out of commits. Include lightweight experiment configurations, metrics, and figures when they belong to the requested work.

## Branch and commit

- Stay on the current feature branch. On `main` or `master`, create a branch such as `feat/yolov10-training` or `docs/experiment-results` before committing. If the base branch has unpublished commits, explain the situation instead of rewriting its history.
- Group related changes by intent. Keep a notebook and its directly related configuration or results together when they form one experiment; separate independent changes.
- Briefly show the proposed messages and file groups, then proceed when the grouping is clear. Multiple commits do not require an extra confirmation.
- Use concise messages in the repository's established style, normally `type(scope): imperative summary`. For example:

  ```text
  feat(data): add parasite dataset conversion
  feat(yolov10): add training notebook
  fix(eval): align class mapping between models
  docs(results): compare detector performance
  chore(deps): add experiment dependencies
  ```

- Stage explicit paths with `git add -- <paths>` and inspect the staged diff before each commit. Avoid blanket staging and preserve intentionally separate staged and unstaged changes.
- This workflow packages existing changes: do not edit files to make them fit a commit. Do not discard work, amend commits, rewrite history, force-push, or bypass hooks. If a hook or Git command fails, report the error and leave the work recoverable.

## Push and open the PR

- Check the final status and new commits. Account for any files left out of scope and ensure the intended changes were included.
- Push the feature branch to its verified upstream, or use `git push --set-upstream origin HEAD` when it has none. Never push directly to `main` or `master`.
- If the push is rejected, report the reason; do not automatically pull, rebase, or force-push.
- Verify the target repository from the remotes. This project's expected repository is `lvckss/parasite-detection-benchmark`; do not reuse the source project's repository or assume that `gh` selected the right target.
- Check for an existing PR for the branch and reuse it. Otherwise, create a PR with explicit repository, base, and head arguments. Determine the actual default branch rather than assuming its name.
- Describe the change and, for experimental work, the configuration and results actually available. Clearly identify any training or evaluation that has not been run. No Issue is required by this skill.
- For multiline PR bodies, write the text to a temporary file and use `gh pr create --body-file <path>`.
- If GitHub CLI or authentication is unavailable, report what succeeded and provide the branch or compare link for manual PR creation.
- Leave approval and merging to the team; do not approve, merge, or enable auto-merge.

## Report

Give the branch, commit hashes and messages, push outcome, PR URL, and any remaining changes or blockers. Keep the report proportional to the work.
