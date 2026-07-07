# Catering Ops Prototype

This is a zero-dependency static prototype for the catering operations
reorientation plan. It is intentionally isolated from the existing MacFleet
Python package so the product direction can be explored without changing the
library code.

## Run

```bash
cd catering-ops-prototype
python3 server.py
```

Open `http://127.0.0.1:8765/today`.

The tiny server falls back to `index.html` for app routes such as `/events`,
`/events/EVT-10041`, `/production`, `/setup-delivery`, `/tasks`, `/review`,
and `/handoff`.

## Prototype Principles

- Events and orders are the primary objects.
- Today is the default surface.
- Event details use a BEO-style sheet structure.
- AI is present as summaries, readiness checks, suggested tasks, and review
  items, but important changes require human approval.
- Data is in-memory seed data shaped for a future API or Postgres-backed
  service.
