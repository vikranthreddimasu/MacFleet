# AI System Design

## Role

AI is an operations aide. It helps staff coordinate, remember, adapt,
summarize, detect risks, prepare decisions, and draft handoff.

## Allowed AI Outputs

- Daily briefing.
- Event readiness summary.
- Missing information detection.
- Change impact summary.
- Suggested tasks.
- Review queue items.
- Handoff draft.

## Disallowed AI Actions

- Directly changing event status.
- Publishing handoff.
- Dismissing risks.
- Creating approved operational tasks.
- Changing customer-visible details.
- Hiding evidence behind a summary.

## Review Requirements

Human review is required for:

- Task creation from AI suggestions.
- Material status changes.
- Risk dismissal.
- Customer-visible notes.
- Published handoff.
- Any operational instruction that changes work.

## Provenance

AI outputs should include:

- Source event.
- Evidence text.
- Affected fields or workflows.
- Suggested action.
- Severity or readiness state.
- Human decision and decision timestamp once reviewed.
