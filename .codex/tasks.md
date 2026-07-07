# Catering Ops Build Tasks

## Completed in this pass

1. Added product direction docs for the catering operations reorientation.
2. Added a zero-dependency static prototype under `catering-ops-prototype/`.
3. Implemented the target navigation: Today, Events, Production, Setup &
   Delivery, Tasks, Review, and Handoff.
4. Added domain seed data for events, BEO-style sections, statuses, checklist
   items, review items, changes, and handoff.
5. Kept AI optional and secondary through readiness summaries, suggested
   tasks, review items, and handoff draft generation.

## Next coding tasks

1. Validate the prototype with real catering users or representative staff.
2. Tune seed data and labels to match the organization's real terminology.
3. Add event editing flows for safe fields.
4. Add persistent local storage for prototype review decisions.
5. Add print/export for event sheets and handoff.
6. Move to a framework app only after the UX structure is accepted.
7. Introduce an API boundary and Postgres schema when persistence is needed.

## Acceptance checklist

- Today is the default route.
- Events and event sheets are understandable without AI context.
- AI suggestions require human approval.
- Review items show evidence and impact.
- Handoff is generated from operational state and remains editable.
