# API Design

## Direction

The API should expose catering operations resources first. AI endpoints should
generate suggestions and drafts, not directly mutate authoritative records.

## Prototype Resources

- Events
- Tasks
- Review items
- Change log
- Handoffs

## Candidate Endpoints

```text
GET    /api/events
GET    /api/events/:id
PATCH  /api/events/:id
GET    /api/tasks
PATCH  /api/tasks/:id
GET    /api/review-items
PATCH  /api/review-items/:id
GET    /api/events/:id/changes
POST   /api/handoffs/generate
PATCH  /api/handoffs/:id
```

## AI Boundary

```text
POST /api/ai/readiness-summary
POST /api/ai/change-impact
POST /api/ai/suggest-tasks
POST /api/ai/handoff-draft
```

Responses should include evidence and suggested actions. The client or API
must save important outputs as pending review records before they affect work.

## Mutation Rule

Only human-initiated API mutations can change:

- Event status.
- Approved checklist items.
- Review decisions.
- Published handoff state.
