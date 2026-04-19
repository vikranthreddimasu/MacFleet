# CLAUDE.md

Project-specific guidance for Claude Code.

## Context

See `AGENTS.md` for the full project tour (architecture, core invariants, module map, key interfaces). That file is the authoritative guide for AI assistants working on MacFleet.

## Critical invariant

**The communication layer never imports torch or mlx.** Gradients flow as numpy arrays between nodes. `macfleet/comm/`, `macfleet/pool/`, and `macfleet/compression/adaptive.py` must stay framework-agnostic. Each engine converts to/from numpy at the boundary.

## Testing

- Framework: pytest + pytest-asyncio (`asyncio_mode = "auto"` in `pyproject.toml`)
- Command: `make test` or `python -m pytest tests/ -v`
- MLX tests auto-skip via `pytest.importorskip("mlx.core")` on non-MLX environments
- Multi-node tests use loopback TCP mesh (`_setup_mesh`, `_make_groups` helpers in `tests/test_comm/test_collectives.py`)

## Lint

`make lint` runs `ruff check` and `mypy --ignore-missing-imports`. Line length 100, ruff selects E/F/W/I.

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke context-save / context-restore
- Code quality, health check → invoke health
