# Frontend Design

## Product Feel

Monochrome, dense, enterprise-grade, and operational. The app should feel like
a modern event book, not a generic AI dashboard.

## Layout System

- Persistent left navigation.
- Sticky top bar with date, search, and status filters.
- Dense tables for schedules, events, setup, and delivery.
- BEO-style event sheet with sticky header and sectioned operational facts.
- Right-side intelligence panel on event detail pages.
- Review queue as a secondary operational workflow.
- Handoff as an editable generated output.

## Navigation

- Today
- Events
- Production
- Setup & Delivery
- Tasks
- Review
- Handoff

Today is the default route. Review and Handoff are important but not primary
entry points.

## UI Patterns

- Use compact status badges for lifecycle, production, setup, delivery,
  readiness, review severity, and task state.
- Use tables where operators need scanning and comparison.
- Use section blocks for event-sheet content.
- Use checklist rows with clear source and approval state.
- Keep AI text in summaries, inline flags, and secondary panels.

## Avoid

- Chat as the center of the app.
- Decorative AI SaaS visuals.
- Oversized metrics as the first screen.
- Abstract risk cards without event context.
- Nested cards and marketing-style hero sections.
