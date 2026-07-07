# Future Architecture

## Near Term

- Keep the prototype static and in-memory until the UX is validated.
- Split data access behind a small repository layer before introducing a
  backend.
- Keep AI mockable and optional.

## Backend Evolution

1. Move seed data into API fixtures.
2. Add a simple service layer for events, tasks, review items, changes, and
   handoffs.
3. Add Postgres tables using the current domain model.
4. Add authentication and role-aware access.
5. Add audit logging for human decisions and AI suggestions.

## AI Evolution

1. Mocked AI output in seed data.
2. Local deterministic suggestion functions for tests.
3. Provider-backed AI service with strict schemas.
4. Human review workflow and provenance storage.
5. Evaluation suite for false positives, missed risks, and handoff quality.

## UI Evolution

- Promote shared components from the static prototype into a framework app
  once the route structure stabilizes.
- Keep Today, Event Detail, and Tasks as the main implementation spine.
- Add printing/export only after the event sheet content stabilizes.
