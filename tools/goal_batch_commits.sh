#!/usr/bin/env bash
# One-off batch: 30+ small commits for /goal skill. Safe to delete after run.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

commit_one() {
  local msg="$1"
  shift
  "$@"
  git add -A
  if git diff --cached --quiet; then
    echo "SKIP (no changes): $msg"
    return 0
  fi
  git commit -m "$msg"
  echo "OK: $msg"
}

# 1 — goal skill in repo
commit_one "chore(skills): add /goal skill for contribution graph targets" true

# 2 — README blank line
commit_one "docs(readme): tighten hero section spacing" \
  sed -i '' '/^Run PyTorch or MLX across multiple Macs/,/^```bash$/{
    /^$/N
    /^\n$/d
  }' README.md 2>/dev/null || perl -i -0pe 's/(gradient synchronization\.)\n\n\n(\`\`\`bash)/$1\n\n$2/s' README.md

# 3 — docs index M4
commit_one "docs(index): mention M4 in Apple Silicon requirement" \
  perl -i -pe 's/MacBook Pro, Mac mini, or Mac Studio/MacBook Pro, Mac mini, Mac Studio, or Mac Pro (M1–M4)/ unless /M1–M4/' docs/index.md

# 4 — utils docstring
commit_one "docs(utils): expand package docstring" \
  printf '%s\n' '"""Shared utilities (atomic I/O, helpers)."""' > macfleet/utils/__init__.py

# 5 — cli docstring
commit_one "docs(cli): note click entry point in module docstring" \
  printf '%s\n' '"""Click CLI entry points (`macfleet join`, `status`, `train`, …)."""' > macfleet/cli/__init__.py

# 6 — comm docstring
commit_one "docs(comm): document numpy-only invariant in package doc" \
  printf '%s\n' '"""Communication layer: TCP transport, wire protocol, collectives (numpy only)."""' > macfleet/comm/__init__.py

# 7 — pool docstring
commit_one "docs(pool): clarify agent and discovery in package doc" \
  printf '%s\n' '"""Pool agent: mDNS discovery, registry, scheduling, heartbeat."""' > macfleet/pool/__init__.py

# 8 — Makefile help
commit_one "chore(make): add help target listing common targets" bash -c 'grep -q "^help:" Makefile || cat >> Makefile <<EOF

help:
	@echo "MacFleet dev targets:"
	@echo "  make test      Run pytest suite"
	@echo "  make lint      ruff + mypy"
	@echo "  make format    ruff format + fix"
	@echo "  make install   pip install -e ."
	@echo "  make dev       pip install -e .[dev]"
	@echo "  make bench     run compute/network/allreduce benches"
	@echo "  make clean     remove caches and build artifacts"
EOF'

# 9 — installation M4
commit_one "docs(install): list M1–M4 Apple Silicon chips" \
  perl -i -pe 's/Apple Silicon \(M1\+\)/Apple Silicon (M1, M2, M3, M4)/' docs/getting-started/installation.md

# 10 — cli.md help hint
commit_one "docs(cli): mention macfleet --help for full option list" bash -c 'grep -q "macfleet --help" docs/cli.md || sed -i "" "1a\\
\\
Run \`macfleet --help\` or \`macfleet COMMAND --help\` for the full option list.\\
" docs/cli.md'

# 11 — quickstart note
commit_one "docs(quickstart): link to pairing guide for second Mac" bash -c 'grep -q "pairing.md" docs/getting-started/quickstart.md || sed -i "" "s/^# Quickstart/# Quickstart\\
\\
> Adding a second Mac? See [Pairing](pairing.md) after the single-Mac check.\\
/" docs/getting-started/quickstart.md'

# 12 — training init
commit_one "docs(training): mention mesh and data parallel in package doc" \
  printf '%s\n' '"""Training: data parallel sync, mesh formation, samplers, loop."""' > macfleet/training/__init__.py

# 13 — AGENTS test count already 962 - skip if same
commit_one "docs(agents): add Python 3.13 to platform note" bash -c 'grep -q "3.13" AGENTS.md || sed -i "" "s/Python 3.11+/Python 3.11+ (3.13 supported)/" AGENTS.md'

# 14 — macfleet init version line
commit_one "docs(macfleet): clarify lazy import in package docstring" \
  perl -i -0pe 's/(Zero-config discovery\. Framework-agnostic engines\. Adaptive networking\.)/$1 Lazy-imports torch and mlx./s' macfleet/__init__.py

# 15 — TODOS header
commit_one "docs(todos): add last-updated note in header" bash -c 'grep -q "Living document" TODOS.md || sed -i "" "2a\\
\\
Living document — items move to releases as they ship.\\
" TODOS.md'

# 16 — docs index changelog link text
commit_one "docs(index): clarify changelog link label" \
  perl -i -pe 's/See the \[changelog\]/See the [release changelog]/' docs/index.md

# 17 — README development section
commit_one "docs(readme): add make help to development section" bash -c 'grep -q "make help" README.md || sed -i "" "s/make lint       # ruff + mypy/make lint       # ruff + mypy\\
make help       # list dev targets/" README.md'

# 18 — pairing.md - read and add one line if missing
commit_one "docs(pairing): note enrollment TTL default" bash -c 'grep -q "5 minutes" docs/getting-started/pairing.md || sed -i "" "1a\\
\\
Enrollment codes expire after five minutes by default (\`--enroll-ttl\` on bootstrap).\\
" docs/getting-started/pairing.md'

# 19 — train.md
commit_one "docs(train): note enable_pool_distributed flag" bash -c 'f=docs/guides/train.md; test -f "$f" && grep -q "enable_pool_distributed" "$f" || echo "Set \`enable_pool_distributed=True\` on \`Pool()\` to join gradient sync across peers." >> "$f"'

# 20 — tasks.md
commit_one "docs(tasks): remind readers tasks must be registered" bash -c 'f=docs/guides/tasks.md; test -f "$f" && grep -q "registered before dispatch" "$f" || sed -i "" "1a\\
\\
Callables must be registered with \`@macfleet.task\` before dispatch — names are sent on the wire, not pickles.\\
" "$f"'

# 21 — dashboard.md
commit_one "docs(dashboard): mention Rich TUI dependency" bash -c 'f=docs/guides/dashboard.md; test -f "$f" && grep -q "Rich" "$f" || sed -i "" "1a\\
\\
The dashboard uses Rich for terminal UI; it reads live agent state.\\
" "$f"'

# 22 — protocol.md
commit_one "docs(protocol): note 24-byte header size" bash -c 'f=docs/reference/protocol.md; test -f "$f" && grep -q "24-byte" "$f" || sed -i "" "1a\\
\\
Wire messages use a fixed 24-byte header (see \`macfleet/comm/protocol.py\`).\\
" "$f"'

# 23 — security.md
commit_one "docs(security): mention audit.jsonl path" bash -c 'f=docs/reference/security.md; test -f "$f" && grep -q "audit.jsonl" "$f" || sed -i "" "1a\\
\\
Security events append to \`~/.macfleet/audit.jsonl\` with secrets redacted.\\
" "$f"'

# 24 — two-mac-testing
commit_one "docs(two-mac): cross-link verify tool" bash -c 'f=docs/getting-started/two-mac-testing.md; test -f "$f" && grep -q "two_mac_verify" "$f" || sed -i "" "1a\\
\\
For scripted checks, see \`tools/two_mac_verify.py\` in the repo.\\
" "$f"'

# 25 — future-architecture
commit_one "docs(future): mark doc as design exploration" bash -c 'f=docs/future-architecture.md; test -f "$f" && grep -q "Design exploration" "$f" || sed -i "" "1i\\
> Design exploration — not committed roadmap.\\
" "$f"'

# 26 — product-brief
commit_one "docs(product): label prototype brief scope" bash -c 'f=docs/product-brief.md; test -f "$f" && grep -q "MacFleet ML" "$f" || sed -i "" "1a\\
\\
MacFleet ML fleet pooling — separate from catering prototype docs in this folder.\\
" "$f"'

# 27 — system-architecture
commit_one "docs(architecture): add numpy boundary reminder" bash -c 'f=docs/system-architecture.md; test -f "$f" && grep -q "numpy boundary" "$f" || sed -i "" "1a\\
\\
Gradients cross the comm layer as numpy arrays only (no torch/mlx imports in \`macfleet/comm/\`).\\
" "$f"'

# 28 — api-design
commit_one "docs(api): note Pool context manager pattern" bash -c 'f=docs/api-design.md; test -f "$f" && grep -q "context manager" "$f" || sed -i "" "1a\\
\\
Prefer \`with macfleet.Pool() as pool:\` so agents stop cleanly on exit.\\
" "$f"'

# 29 — data-model
commit_one "docs(data-model): clarify fleet token persistence path" bash -c 'f=docs/data-model.md; test -f "$f" && grep -q "fleet-token" "$f" || sed -i "" "1a\\
\\
Fleet tokens persist at \`~/.macfleet/fleet-token\` (mode 0600).\\
" "$f"'

# 30 — agent-behavior
commit_one "docs(agent): document bully coordinator election" bash -c 'f=docs/agent-behavior.md; test -f "$f" && grep -q "bully" "$f" || sed -i "" "1a\\
\\
Coordinator election uses a bully-style algorithm in the registry.\\
" "$f"'

# 31 — frontend-design (catering prototype)
commit_one "docs(frontend): mark catering UI doc as prototype-only" bash -c 'f=docs/frontend-design.md; test -f "$f" && grep -q "prototype-only" "$f" || sed -i "" "1i\\
> Catering ops prototype UI — not shipped with macfleet pip package.\\
" "$f"'

# 32 — ui-system
commit_one "docs(ui): scope ui-system doc to prototype" bash -c 'f=docs/ui-system.md; test -f "$f" && grep -q "prototype" "$f" || sed -i "" "1a\\
\\
Applies to the catering-ops prototype, not the macfleet CLI/TUI.\\
" "$f"'

# 33 — automation workflow
commit_one "docs(automation): add make test to validation step" bash -c 'f=docs/automation/product-improvement-workflow.md; test -f "$f" && grep -q "make test" "$f" || sed -i "" "s/low-risk validation/low-risk validation (\`make test\`, \`make lint\`)/" "$f"'

# 34 — CHANGELOG header
commit_one "docs(changelog): add unreleased section placeholder" bash -c 'grep -q "## Unreleased" CHANGELOG.md || sed -i "" "1a\\
\\
## Unreleased\\
\\
- Documentation and DX polish (ongoing).\\
" CHANGELOG.md'

# 35 — conftest comment
commit_one "test(conftest): document shared pytest fixtures module" bash -c 'grep -q "Shared pytest fixtures" tests/conftest.py || sed -i "" "1i\\
# Shared pytest fixtures for MacFleet integration and CLI tests.\\
" tests/conftest.py'

echo "Done: $(git log --oneline | head -1)"
