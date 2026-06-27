# Product-improvement workflow

MacFleet improvements should be small, useful, and safe to merge. This runbook
turns the daily scheduled audit into a focused engineering pass without creating
fake commits or handing GitHub Actions credentials the ability to rewrite code.

## Daily schedule

The Codex recurring task should run every day at **3:00 AM America/New_York**
against the MacFleet checkout. It should start from the latest `master`, create a
human-named branch for the chosen improvement, make only meaningful commits, run
checks, push the branch, open a PR, and merge only when the safety criteria below
are met.

`.github/workflows/product-improvement.yml` is a safe GitHub Actions fallback
audit. It can be started with **Run workflow** in GitHub Actions and is scheduled
at the UTC offsets that correspond to 3:00 AM America/New_York. A local-time gate
keeps the daily scheduled run aligned with daylight saving time. The workflow
checks out the repository, runs low-risk validation, scans for TODOs and
security-sensitive areas, and uploads a `product-improvement-report` artifact.

The scheduled workflow intentionally has `contents: read` only. It does not push
branches or open pull requests by itself, because automated code changes and
auto-merges need human or agent review to avoid unsafe edits, leaked secrets, and
unreviewed auth/security regressions.

## Manual improvement pass

Use this checklist after reviewing the latest `product-improvement-report`.

1. Fetch the latest repository state.

   ```bash
   git fetch --all --prune
   git status --short --branch
   git switch master
   git pull --ff-only origin master
   ```

2. Confirm the working tree is clean.

   ```bash
   git status --short
   ```

3. Pick the highest-impact weakness from the report and recent code review.
   Prefer security risk, data loss/corruption risk, broken core flows,
   reliability, validation, error handling, or missing tests before docs-only
   work.

4. Create a human-readable branch name based on the work.

   ```bash
   git switch -c improve-auth-errors
   ```

5. Implement one meaningful unit of work at a time. For each unit:

   - Explain why it matters in the PR description.
   - Add or update tests when practical.
   - Run the relevant targeted check before committing.
   - Commit with a clear message such as `Add tests for auth failure paths`.

6. Run broader validation before opening the PR.

   ```bash
   ruff check macfleet/ tests/
   python -m pytest tests/ -v
   ```

7. Scan for accidental secrets before pushing.

   ```bash
   git diff --check master...HEAD
   rg -n "(BEGIN .*PRIVATE KEY|AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN|password=|token=|secret=)" . --glob '!docs/automation/product-improvement-workflow.md'
   ```

8. Push and open a PR into `master`.

   ```bash
   git push -u origin HEAD
   gh pr create --base master --fill
   ```

9. Merge only when checks pass, there are no conflicts, no secrets are present,
   and the changes are safe. Leave the PR open for manual review if it touches
   authentication, authorization, credential handling, destructive actions,
   deployment secrets, or another high-risk area.

10. After a safe merge, delete the branch locally and remotely.

    ```bash
    gh pr merge --squash --delete-branch
    git switch master
    git pull --ff-only origin master
    ```

## PR description template

```markdown
## Summary
-

## Security changes
-

## Usability changes
-

## Tests/checks run
-

## Commits
-

## Risks / manual review notes
-

## Merge status
- Not merged yet / merged after checks passed
```
