# System Architecture: Catering Ops Prototype

Gradients cross the comm layer as numpy arrays only (no torch/mlx imports in `macfleet/comm/`).


## Prototype Shape

The prototype is intentionally simple:

- Static frontend shell.
- In-memory seed data.
- Client-side routing for `/today`, `/events`, `/events/:id`, `/production`,
  `/setup-delivery`, `/tasks`, `/review`, and `/handoff`.
- Mock AI suggestions stored as review items, readiness summaries, and
  suggested tasks.

This gives the product a concrete UX before committing to backend complexity.

## Domain Boundary

Domain-first resources:

- Event
- ChecklistItem
- ReviewItem
- ChangeLog
- HandoffDraft

AI is not a primary resource owner. It returns suggestions and summaries that
are attached to domain records.

## Future Service Boundary

When persistence is added, use a thin API layer with clear resources:

- `GET /api/events`
- `GET /api/events/:id`
- `PATCH /api/events/:id`
- `GET /api/tasks`
- `PATCH /api/tasks/:id`
- `GET /api/review-items`
- `PATCH /api/review-items/:id`
- `POST /api/handoffs/generate`
- `PATCH /api/handoffs/:id`

AI calls should run behind a mockable service boundary. AI outputs must be
stored with provenance and approval state.

## Human Control Rule

AI may draft, detect, summarize, and suggest. It must not directly mutate
event records, publish handoffs, dismiss risks, or create operational work
without human review.
