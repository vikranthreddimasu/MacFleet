# Prototype Scope

## In Scope

- Daily schedule as the default first screen.
- Searchable event/order list.
- BEO-style event detail pages.
- Production, setup, delivery, task, review, and handoff workflows.
- In-memory seed data shaped for future API and Postgres storage.
- Mockable AI outputs: readiness summary, missing information, change impact,
  suggested tasks, review items, and handoff draft.
- Human approval controls for suggested tasks and review decisions.

## Out of Scope

- Invoicing, payments, pricing, deposits, and tax.
- Customer portal or CRM.
- Full menu catalog management.
- Inventory purchasing.
- Real authentication and authorization.
- Real AI calls.
- Real Postgres persistence.
- Multi-tenant deployment.

## Current Prototype Location

The current working prototype lives at `catering-ops-prototype/`. It is a
zero-dependency static app with a small Python history-fallback server.

## Acceptance Scenarios

- A manager opens Today and sees the service schedule first.
- A coordinator opens an event sheet and sees familiar BEO-style sections.
- A kitchen lead sees production tasks and dietary changes.
- A setup lead sees load times, delivery times, access notes, and setup
  blockers.
- An AI-suggested task remains pending until a human approves it.
- Review items show evidence, impact, and suggested actions.
- Handoff is generated from operational state and remains editable before
  sharing.
