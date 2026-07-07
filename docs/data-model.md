# Data Model

## Event

Minimum prototype fields:

- `id`
- `orderNumber`
- `eventName`
- `eventType`
- `serviceType`
- `date`
- `startTime`
- `endTime`
- `loadTime`
- `deliveryTime`
- `guestCount`
- `coordinator`
- `lifecycleStatus`
- `productionStatus`
- `setupStatus`
- `deliveryStatus`
- `readiness`
- `readinessSummary`
- `customer`
- `location`
- `menu`
- `dietary`
- `equipment`
- `staffing`
- `setupNotes`
- `deliveryNotes`
- `internalNotes`
- `aiSummary`

## ChecklistItem

- `id`
- `eventId`
- `phase`
- `title`
- `ownerRole`
- `dueTime`
- `status`
- `source`
- `approvalState`
- `suggestedBecause`

## ReviewItem

- `id`
- `eventId`
- `type`
- `severity`
- `status`
- `title`
- `evidence`
- `impact`
- `suggestedAction`
- `humanDecision`

## ChangeLog

- `id`
- `eventId`
- `timestamp`
- `actor`
- `source`
- `field`
- `before`
- `after`

## HandoffDraft

- `id`
- `date`
- `body`
- `sourceEventIds`
- `sourceReviewItemIds`
- `sourceTaskIds`
- `status`
- `reviewedBy`
- `reviewedAt`

## Persistence Note

The prototype uses in-memory seed data. The model is deliberately shaped so it
can become normalized Postgres tables later.
