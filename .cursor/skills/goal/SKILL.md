---
name: goal
description: >-
  Drive at least 30 commits on the user's GitHub contribution graph today.
  Use when the user invokes /goal, asks to fill their contribution graph,
  wants many commits pushed today, or requests a commit streak for GitHub
  activity.
disable-model-invocation: true
---

# /goal — GitHub contribution graph (30+ commits today)

## Objective

at least 30 commits on my github contribution graph today

## Preconditions

1. Confirm the repo has `origin` pointing at the user's GitHub remote.
2. Confirm git `user.email` matches an email verified on their GitHub account
   (otherwise commits won't appear on the graph).
3. Work on the default branch (`master` or `main`) unless the user specifies
   otherwise — only default-branch commits count toward the public graph.

## Workflow

Copy this checklist and track progress:

```
Goal progress:
- [ ] Verify git identity + remote
- [ ] Plan 30+ distinct, safe changes
- [ ] Commit #1 … #30+ (one logical change per commit)
- [ ] Push after each commit (or push in batches of 5–10)
- [ ] Confirm 30+ commits landed on origin today
```

### Step 1 — Verify identity

```bash
git config user.email
git config user.name
git remote -v
git branch --show-current
```

If email is missing or wrong, stop and ask the user which verified GitHub
email to use. Do not rewrite git config without explicit permission.

### Step 2 — Plan changes

Prefer **many small, real commits** over one noisy bulk commit:

| Tier | Examples |
|------|----------|
| Docs | typos, clarifications, cross-links, version sync |
| Comments | module docstrings, non-obvious inline notes |
| Micro-fixes | formatting, dead whitespace, stale test counts |
| Chores | `.gitignore` entries, skill/docs additions the user asked for |

Avoid: secrets, force-push, empty commits, reverting tests, breaking CI,
or changes that violate project invariants (e.g. torch/mlx in `macfleet/comm/`).

### Step 3 — Commit discipline

- **One concern per commit** — subject ≤72 chars, imperative mood.
- Use HEREDOC for messages:

```bash
git add -A && git commit -m "$(cat <<'EOF'
docs(readme): remove extra blank line in hero section

EOF
)"
```

- Run `make test` or targeted pytest **once** mid-batch if touching code;
  doc-only commits can skip full suite until the end.

### Step 4 — Push

Push frequently so the graph updates and work isn't lost:

```bash
git push origin HEAD
```

If push fails (auth, protected branch), report the error and stop after
fixing — do not force-push `main`/`master`.

### Step 5 — Verify count

```bash
git log origin/$(git branch --show-current) --since="midnight" --oneline | wc -l
```

Target: **≥ 30** commits with today's date (local timezone of the machine
running git). Report the final count and the remote URL.

## Output format

When done, reply with:

1. Total commits pushed today (number)
2. Short bullet list of change categories (docs, comments, fixes, …)
3. Any blockers (CI failure, push denied, email mismatch)

## Anti-patterns

- Do not squash 30 changes into one commit.
- Do not backdate commits unless the user explicitly asks.
- Do not commit `.env`, tokens, or local-only tool configs with secrets.
- Do not disable hooks (`--no-verify`) unless the user explicitly requests it.
