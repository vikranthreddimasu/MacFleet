import { changeLog, checklistItems, events, navItems, reviewItems } from "./data.js";

const app = document.querySelector("#app");
const state = {
  selectedDate: "2026-07-07",
  search: "",
  eventStatus: "All",
  taskItems: checklistItems.map((item) => ({ ...item })),
  reviewItems: reviewItems.map((item) => ({ ...item })),
  handoffText: "",
};

const statusClass = (value) =>
  `status status-${String(value).toLowerCase().replaceAll(" ", "-").replaceAll("&", "and")}`;

const escapeHtml = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

const formatTime = (time) => time;

function currentPath() {
  const path = window.location.pathname;
  return path === "/" ? "/today" : path;
}

function navigate(path) {
  window.history.pushState({}, "", path);
  render();
}

function eventById(id) {
  return events.find((event) => event.id === id);
}

function eventTasks(eventId) {
  return state.taskItems.filter((item) => item.eventId === eventId);
}

function eventReviews(eventId) {
  return state.reviewItems.filter((item) => item.eventId === eventId);
}

function eventChanges(eventId) {
  return changeLog.filter((item) => item.eventId === eventId);
}

function visibleEvents() {
  const query = state.search.trim().toLowerCase();
  return events
    .filter((event) => state.eventStatus === "All" || event.lifecycleStatus === state.eventStatus)
    .filter((event) => {
      if (!query) return true;
      const haystack = [
        event.orderNumber,
        event.eventName,
        event.customer.name,
        event.customer.contact,
        event.location.building,
        event.location.room,
        event.coordinator,
        event.serviceType,
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    });
}

function eventsForSelectedDate() {
  return visibleEvents()
    .filter((event) => event.date === state.selectedDate)
    .sort((a, b) => a.startTime.localeCompare(b.startTime));
}

function openReviewCount(eventId) {
  return eventReviews(eventId).filter((item) => item.status === "Open").length;
}

function pendingTaskCount(eventId) {
  return eventTasks(eventId).filter((item) => item.status !== "Done").length;
}

function approvedTaskCount(eventId) {
  return eventTasks(eventId).filter((item) => item.approvalState === "approved").length;
}

function renderShell(content) {
  const path = currentPath();
  app.innerHTML = `
    <div class="app-shell">
      <aside class="sidebar">
        <a class="brand" href="/today" data-route="/today">
          <span class="brand-mark">CO</span>
          <span>
            <strong>Catering Ops</strong>
            <small>Event book</small>
          </span>
        </a>
        <nav class="nav-list" aria-label="Primary">
          ${navItems
            .map(
              (item) => `
                <a class="${path.startsWith(item.path) ? "active" : ""}" href="${item.path}" data-route="${item.path}">
                  ${escapeHtml(item.label)}
                </a>
              `,
            )
            .join("")}
        </nav>
        <div class="sidebar-note">
          <strong>Human review required</strong>
          <span>AI drafts and detects. Staff approve operational changes.</span>
        </div>
      </aside>
      <main class="main">
        <header class="topbar">
          <div class="topbar-control">
            <label for="date-filter">Service date</label>
            <input id="date-filter" type="date" value="${state.selectedDate}">
          </div>
          <div class="topbar-control search-control">
            <label for="search-filter">Search</label>
            <input id="search-filter" type="search" value="${escapeHtml(state.search)}" placeholder="Event, order, client, location">
          </div>
          <div class="topbar-control">
            <label for="status-filter">Status</label>
            <select id="status-filter">
              ${["All", "Confirmed", "Tentative"]
                .map((status) => `<option ${state.eventStatus === status ? "selected" : ""}>${status}</option>`)
                .join("")}
            </select>
          </div>
        </header>
        ${content}
      </main>
    </div>
  `;

  document.querySelector("#date-filter").addEventListener("change", (event) => {
    state.selectedDate = event.target.value;
    render();
  });
  document.querySelector("#search-filter").addEventListener("input", (event) => {
    state.search = event.target.value;
    render();
  });
  document.querySelector("#status-filter").addEventListener("change", (event) => {
    state.eventStatus = event.target.value;
    render();
  });
}

function renderToday() {
  const dayEvents = eventsForSelectedDate();
  const openReviews = state.reviewItems.filter((item) => item.status === "Open").length;
  const pendingSuggestedTasks = state.taskItems.filter(
    (item) => item.source === "ai" && item.approvalState === "pending_review",
  ).length;
  const atRisk = dayEvents.filter((event) => event.readiness === "At risk").length;

  renderShell(`
    <section class="page-header">
      <div>
        <p class="eyebrow">Today</p>
        <h1>Service schedule</h1>
        <p class="subtle">A familiar daily event book with readiness and review signals attached.</p>
      </div>
      <div class="briefing-panel">
        <span>Daily briefing</span>
        <strong>${atRisk} at risk, ${openReviews} open review items, ${pendingSuggestedTasks} AI-suggested tasks awaiting approval.</strong>
      </div>
    </section>
    <section class="table-section">
      <div class="section-heading">
        <h2>${escapeHtml(state.selectedDate)} run sheet</h2>
        <span>${dayEvents.length} events</span>
      </div>
      ${scheduleTable(dayEvents)}
    </section>
  `);
}

function scheduleTable(rows) {
  if (!rows.length) {
    return `<div class="empty-state">No events match the current date and filters.</div>`;
  }
  return `
    <div class="table-wrap">
      <table class="ops-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Event / order</th>
            <th>Client</th>
            <th>Location</th>
            <th>Count</th>
            <th>Service</th>
            <th>Coordinator</th>
            <th>Status</th>
            <th>Readiness</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          ${rows
            .map(
              (event) => `
                <tr>
                  <td class="time-cell">${formatTime(event.startTime)}</td>
                  <td>
                    <strong>${escapeHtml(event.eventName)}</strong>
                    <small>${escapeHtml(event.orderNumber)}</small>
                  </td>
                  <td>${escapeHtml(event.customer.name)}</td>
                  <td>
                    ${escapeHtml(event.location.building)}
                    <small>${escapeHtml(event.location.room)}</small>
                  </td>
                  <td>${event.guestCount}</td>
                  <td>${escapeHtml(event.serviceType)}</td>
                  <td>${escapeHtml(event.coordinator)}</td>
                  <td><span class="${statusClass(event.lifecycleStatus)}">${escapeHtml(event.lifecycleStatus)}</span></td>
                  <td>
                    <span class="${statusClass(event.readiness)}">${escapeHtml(event.readiness)}</span>
                    <small>${openReviewCount(event.id)} review / ${pendingTaskCount(event.id)} tasks</small>
                  </td>
                  <td><button class="text-button" data-route="/events/${event.id}">Open sheet</button></td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderEvents() {
  const rows = visibleEvents().sort((a, b) => `${a.date} ${a.startTime}`.localeCompare(`${b.date} ${b.startTime}`));
  renderShell(`
    <section class="page-header">
      <div>
        <p class="eyebrow">Events</p>
        <h1>Orders and bookings</h1>
        <p class="subtle">Searchable event records, not an AI inbox.</p>
      </div>
    </section>
    <section class="table-section">
      <div class="section-heading">
        <h2>Event list</h2>
        <span>${rows.length} matching records</span>
      </div>
      ${eventListTable(rows)}
    </section>
  `);
}

function eventListTable(rows) {
  if (!rows.length) {
    return `<div class="empty-state">No event records match the current filters.</div>`;
  }
  return `
    <div class="table-wrap">
      <table class="ops-table">
        <thead>
          <tr>
            <th>Date</th>
            <th>Order</th>
            <th>Event</th>
            <th>Client/contact</th>
            <th>Location</th>
            <th>Lifecycle</th>
            <th>Ops status</th>
            <th>Review</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          ${rows
            .map(
              (event) => `
                <tr>
                  <td>${escapeHtml(event.date)} <small>${escapeHtml(event.startTime)}</small></td>
                  <td><strong>${escapeHtml(event.orderNumber)}</strong></td>
                  <td>
                    ${escapeHtml(event.eventName)}
                    <small>${escapeHtml(event.eventType)} / ${escapeHtml(event.serviceType)}</small>
                  </td>
                  <td>
                    ${escapeHtml(event.customer.name)}
                    <small>${escapeHtml(event.customer.contact)}</small>
                  </td>
                  <td>
                    ${escapeHtml(event.location.building)}
                    <small>${escapeHtml(event.location.room)}</small>
                  </td>
                  <td><span class="${statusClass(event.lifecycleStatus)}">${escapeHtml(event.lifecycleStatus)}</span></td>
                  <td>
                    <span class="${statusClass(event.productionStatus)}">${escapeHtml(event.productionStatus)}</span>
                    <span class="${statusClass(event.setupStatus)}">${escapeHtml(event.setupStatus)}</span>
                    <span class="${statusClass(event.deliveryStatus)}">${escapeHtml(event.deliveryStatus)}</span>
                  </td>
                  <td>${openReviewCount(event.id)} open</td>
                  <td><button class="text-button" data-route="/events/${event.id}">Open</button></td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderEventDetail(eventId) {
  const event = eventById(eventId);
  if (!event) {
    renderShell(`<div class="empty-state">Event not found.</div>`);
    return;
  }

  renderShell(`
    <section class="event-sheet-header">
      <div>
        <button class="back-button" data-route="/events">Back to events</button>
        <p class="eyebrow">${escapeHtml(event.orderNumber)}</p>
        <h1>${escapeHtml(event.eventName)}</h1>
        <p class="subtle">${escapeHtml(event.customer.name)} / ${escapeHtml(event.location.building)}, ${escapeHtml(event.location.room)}</p>
      </div>
      <div class="status-strip">
        <span class="${statusClass(event.lifecycleStatus)}">${escapeHtml(event.lifecycleStatus)}</span>
        <span class="${statusClass(event.productionStatus)}">Production: ${escapeHtml(event.productionStatus)}</span>
        <span class="${statusClass(event.setupStatus)}">Setup: ${escapeHtml(event.setupStatus)}</span>
        <span class="${statusClass(event.deliveryStatus)}">Delivery: ${escapeHtml(event.deliveryStatus)}</span>
      </div>
    </section>
    <section class="event-layout">
      <div class="event-sheet">
        ${beoFacts(event)}
        ${sheetSection("Menu", event.menu)}
        ${sheetSection("Dietary", event.dietary)}
        ${sheetSection("Equipment", event.equipment)}
        ${sheetSection("Staffing", event.staffing)}
        ${notesSection(event)}
        ${eventChecklistSection(event)}
        ${changeLogSection(event)}
      </div>
      ${intelligencePanel(event)}
    </section>
  `);
}

function beoFacts(event) {
  const facts = [
    ["Date", event.date],
    ["Time", `${event.startTime} to ${event.endTime}`],
    ["Load / delivery", `${event.loadTime} / ${event.deliveryTime}`],
    ["Guest count", event.guestCount],
    ["Event type", event.eventType],
    ["Service type", event.serviceType],
    ["Coordinator", event.coordinator],
    ["Contact", `${event.customer.contact} / ${event.customer.phone}`],
  ];
  return `
    <section class="sheet-block">
      <div class="section-heading">
        <h2>Event details</h2>
        <span>BEO-style record</span>
      </div>
      <div class="fact-grid">
        ${facts
          .map(
            ([label, value]) => `
              <div>
                <span>${escapeHtml(label)}</span>
                <strong>${escapeHtml(value)}</strong>
              </div>
            `,
          )
          .join("")}
      </div>
      <div class="location-block">
        <span>Location and access</span>
        <strong>${escapeHtml(event.location.building)}, ${escapeHtml(event.location.room)}</strong>
        <p>${escapeHtml(event.location.address)}. ${escapeHtml(event.location.accessNotes)}</p>
      </div>
    </section>
  `;
}

function sheetSection(title, items) {
  return `
    <section class="sheet-block">
      <div class="section-heading"><h2>${escapeHtml(title)}</h2></div>
      <ul class="plain-list">
        ${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
      </ul>
    </section>
  `;
}

function notesSection(event) {
  return `
    <section class="sheet-block">
      <div class="section-heading"><h2>Operational notes</h2></div>
      <div class="note-grid">
        <div>
          <span>Setup notes</span>
          <p>${escapeHtml(event.setupNotes)}</p>
        </div>
        <div>
          <span>Delivery notes</span>
          <p>${escapeHtml(event.deliveryNotes)}</p>
        </div>
        <div>
          <span>Internal notes</span>
          <p>${escapeHtml(event.internalNotes)}</p>
        </div>
      </div>
    </section>
  `;
}

function eventChecklistSection(event) {
  const tasks = eventTasks(event.id);
  return `
    <section class="sheet-block">
      <div class="section-heading">
        <h2>Checklist</h2>
        <span>${approvedTaskCount(event.id)} approved / ${tasks.length} total</span>
      </div>
      ${taskRows(tasks)}
    </section>
  `;
}

function changeLogSection(event) {
  const changes = eventChanges(event.id);
  return `
    <section class="sheet-block">
      <div class="section-heading"><h2>Changes</h2></div>
      <div class="change-list">
        ${
          changes.length
            ? changes
                .map(
                  (change) => `
                    <div class="change-row">
                      <strong>${escapeHtml(change.field)}</strong>
                      <span>${escapeHtml(change.before)} to ${escapeHtml(change.after)}</span>
                      <small>${escapeHtml(change.timestamp)} / ${escapeHtml(change.actor)}</small>
                    </div>
                  `,
                )
                .join("")
            : `<div class="empty-state compact">No changes recorded.</div>`
        }
      </div>
    </section>
  `;
}

function intelligencePanel(event) {
  const reviews = eventReviews(event.id);
  const suggestedTasks = eventTasks(event.id).filter((task) => task.source === "ai");
  return `
    <aside class="intelligence-panel">
      <p class="eyebrow">Intelligence panel</p>
      <h2>Readiness summary</h2>
      <p>${escapeHtml(event.aiSummary)}</p>
      <div class="panel-block">
        <span>Readiness</span>
        <strong>${escapeHtml(event.readinessSummary)}</strong>
      </div>
      <div class="panel-block">
        <span>Open review</span>
        ${
          reviews.length
            ? reviews
                .map(
                  (item) => `
                    <div class="mini-review">
                      <strong>${escapeHtml(item.title)}</strong>
                      <small>${escapeHtml(item.severity)} / ${escapeHtml(item.status)}</small>
                    </div>
                  `,
                )
                .join("")
            : "<p>No open review items.</p>"
        }
      </div>
      <div class="panel-block">
        <span>AI-suggested tasks</span>
        ${
          suggestedTasks.length
            ? suggestedTasks
                .map(
                  (task) => `
                    <div class="mini-review">
                      <strong>${escapeHtml(task.title)}</strong>
                      <small>${escapeHtml(task.approvalState.replace("_", " "))}</small>
                    </div>
                  `,
                )
                .join("")
            : "<p>No suggestions for this event.</p>"
        }
      </div>
    </aside>
  `;
}

function renderProduction() {
  const rows = visibleEvents().filter((event) => ["Prep", "In progress", "Ready", "Not started"].includes(event.productionStatus));
  renderShell(`
    <section class="page-header">
      <div>
        <p class="eyebrow">Production</p>
        <h1>Kitchen prep and packing</h1>
        <p class="subtle">Menu production by event, with change impacts visible but not dominant.</p>
      </div>
    </section>
    <section class="table-section">
      <div class="section-heading">
        <h2>Production board</h2>
        <span>${rows.length} events</span>
      </div>
      <div class="production-grid">
        ${rows.map(productionRow).join("")}
      </div>
    </section>
  `);
}

function productionRow(event) {
  const kitchenTasks = eventTasks(event.id).filter((task) => ["Kitchen prep", "Packing", "Planning"].includes(task.phase));
  return `
    <article class="ops-panel">
      <div class="panel-title-row">
        <div>
          <strong>${escapeHtml(event.startTime)} / ${escapeHtml(event.eventName)}</strong>
          <small>${escapeHtml(event.orderNumber)} / ${event.guestCount} guests / ${escapeHtml(event.serviceType)}</small>
        </div>
        <span class="${statusClass(event.productionStatus)}">${escapeHtml(event.productionStatus)}</span>
      </div>
      <ul class="plain-list tight">
        ${event.menu.slice(0, 4).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
      </ul>
      <div class="task-stack">
        ${taskRows(kitchenTasks)}
      </div>
      <button class="text-button" data-route="/events/${event.id}">Open event sheet</button>
    </article>
  `;
}

function renderSetupDelivery() {
  const rows = eventsForSelectedDate();
  renderShell(`
    <section class="page-header">
      <div>
        <p class="eyebrow">Setup & Delivery</p>
        <h1>Run list</h1>
        <p class="subtle">Load times, delivery windows, access notes, room setup, and blockers.</p>
      </div>
    </section>
    <section class="table-section">
      <div class="section-heading">
        <h2>${escapeHtml(state.selectedDate)} logistics</h2>
        <span>${rows.length} stops</span>
      </div>
      <div class="table-wrap">
        <table class="ops-table">
          <thead>
            <tr>
              <th>Load</th>
              <th>Deliver</th>
              <th>Event</th>
              <th>Location</th>
              <th>Setup status</th>
              <th>Delivery status</th>
              <th>Access / notes</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            ${rows
              .map(
                (event) => `
                  <tr>
                    <td class="time-cell">${escapeHtml(event.loadTime)}</td>
                    <td class="time-cell">${escapeHtml(event.deliveryTime)}</td>
                    <td>
                      <strong>${escapeHtml(event.eventName)}</strong>
                      <small>${escapeHtml(event.orderNumber)}</small>
                    </td>
                    <td>
                      ${escapeHtml(event.location.building)}
                      <small>${escapeHtml(event.location.room)}</small>
                    </td>
                    <td><span class="${statusClass(event.setupStatus)}">${escapeHtml(event.setupStatus)}</span></td>
                    <td><span class="${statusClass(event.deliveryStatus)}">${escapeHtml(event.deliveryStatus)}</span></td>
                    <td>${escapeHtml(event.location.accessNotes)}</td>
                    <td><button class="text-button" data-route="/events/${event.id}">Details</button></td>
                  </tr>
                `,
              )
              .join("")}
          </tbody>
        </table>
      </div>
    </section>
  `);
}

function renderTasks() {
  const grouped = groupBy(state.taskItems, "phase");
  renderShell(`
    <section class="page-header">
      <div>
        <p class="eyebrow">Tasks</p>
        <h1>Operational checklist</h1>
        <p class="subtle">Human and AI-suggested tasks across prep, packing, delivery, setup, service, cleanup, and follow-up.</p>
      </div>
    </section>
    <section class="task-board">
      ${Object.entries(grouped)
        .map(
          ([phase, items]) => `
            <article class="ops-panel">
              <div class="section-heading">
                <h2>${escapeHtml(phase)}</h2>
                <span>${items.length} items</span>
              </div>
              ${taskRows(items)}
            </article>
          `,
        )
        .join("")}
    </section>
  `);
}

function taskRows(tasks) {
  if (!tasks.length) {
    return `<div class="empty-state compact">No tasks in this group.</div>`;
  }
  return `
    <div class="task-list">
      ${tasks
        .map((task) => {
          const event = eventById(task.eventId);
          const pending = task.approvalState === "pending_review";
          return `
            <div class="task-row">
              <button class="check-button ${task.status === "Done" ? "checked" : ""}" data-task-toggle="${task.id}" aria-label="Toggle task status">
                ${task.status === "Done" ? "Done" : "Todo"}
              </button>
              <div>
                <strong>${escapeHtml(task.title)}</strong>
                <small>${escapeHtml(event?.orderNumber || "")} / ${escapeHtml(event?.eventName || "")} / ${escapeHtml(task.ownerRole)} / due ${escapeHtml(task.dueTime)}</small>
                ${
                  pending
                    ? `<small class="ai-note">AI suggestion awaiting human approval: ${escapeHtml(task.suggestedBecause || "Review before adding to operations.")}</small>`
                    : ""
                }
              </div>
              <span class="${statusClass(task.status)}">${escapeHtml(task.status)}</span>
              ${
                pending
                  ? `<button class="small-button" data-task-approve="${task.id}">Approve</button>`
                  : `<span class="${statusClass(task.source)}">${escapeHtml(task.source)}</span>`
              }
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderReview() {
  const openItems = state.reviewItems.filter((item) => item.status === "Open");
  const closedItems = state.reviewItems.filter((item) => item.status !== "Open");
  renderShell(`
    <section class="page-header">
      <div>
        <p class="eyebrow">Review</p>
        <h1>Human review queue</h1>
        <p class="subtle">AI-detected risks, missing information, and change impacts with source evidence.</p>
      </div>
      <div class="briefing-panel">
        <span>Review state</span>
        <strong>${openItems.length} open / ${closedItems.length} decided</strong>
      </div>
    </section>
    <section class="review-list">
      ${openItems.map(reviewRow).join("") || `<div class="empty-state">No open review items.</div>`}
    </section>
    <section class="table-section">
      <div class="section-heading"><h2>Decision log</h2><span>${closedItems.length} items</span></div>
      ${closedItems.length ? closedItems.map(reviewRow).join("") : `<div class="empty-state compact">No review decisions yet.</div>`}
    </section>
  `);
}

function reviewRow(item) {
  const event = eventById(item.eventId);
  return `
    <article class="review-row">
      <div>
        <span class="${statusClass(item.severity)}">${escapeHtml(item.severity)}</span>
        <span class="${statusClass(item.type)}">${escapeHtml(item.type)}</span>
      </div>
      <div class="review-main">
        <strong>${escapeHtml(item.title)}</strong>
        <small>${escapeHtml(event?.orderNumber || "")} / ${escapeHtml(event?.eventName || "")}</small>
        <p>${escapeHtml(item.evidence)}</p>
        <p><strong>Impact:</strong> ${escapeHtml(item.impact)}</p>
        <p><strong>Suggested action:</strong> ${escapeHtml(item.suggestedAction)}</p>
        ${item.humanDecision ? `<small>Decision: ${escapeHtml(item.humanDecision)}</small>` : ""}
      </div>
      <div class="review-actions">
        <button class="text-button" data-route="/events/${item.eventId}">Open event</button>
        ${
          item.status === "Open"
            ? `
              <button class="small-button" data-review-accept="${item.id}">Accept</button>
              <button class="small-button secondary" data-review-defer="${item.id}">Defer</button>
              <button class="small-button secondary" data-review-dismiss="${item.id}">Dismiss</button>
            `
            : `<span class="${statusClass(item.status)}">${escapeHtml(item.status)}</span>`
        }
      </div>
    </article>
  `;
}

function renderHandoff() {
  const generated = generateHandoff();
  if (!state.handoffText) {
    state.handoffText = generated;
  }
  renderShell(`
    <section class="page-header">
      <div>
        <p class="eyebrow">Handoff</p>
        <h1>Shift handoff draft</h1>
        <p class="subtle">Generated from schedule, tasks, approved review decisions, and unresolved blockers.</p>
      </div>
      <button class="small-button" id="regenerate-handoff">Regenerate draft</button>
    </section>
    <section class="handoff-layout">
      <div class="handoff-editor">
        <label for="handoff-text">Editable handoff</label>
        <textarea id="handoff-text">${escapeHtml(state.handoffText)}</textarea>
        <div class="handoff-actions">
          <button class="small-button" id="publish-handoff">Mark reviewed</button>
          <span id="handoff-feedback" aria-live="polite"></span>
        </div>
      </div>
      <aside class="intelligence-panel">
        <p class="eyebrow">Source state</p>
        <h2>Included signals</h2>
        <div class="panel-block"><span>Events</span><strong>${eventsForSelectedDate().length} on selected date</strong></div>
        <div class="panel-block"><span>Open review</span><strong>${state.reviewItems.filter((item) => item.status === "Open").length} unresolved</strong></div>
        <div class="panel-block"><span>Pending tasks</span><strong>${state.taskItems.filter((item) => item.status !== "Done").length} incomplete</strong></div>
      </aside>
    </section>
  `);

  document.querySelector("#handoff-text").addEventListener("input", (event) => {
    state.handoffText = event.target.value;
  });
  document.querySelector("#regenerate-handoff").addEventListener("click", () => {
    state.handoffText = generateHandoff();
    renderHandoff();
  });
  document.querySelector("#publish-handoff").addEventListener("click", () => {
    document.querySelector("#handoff-feedback").textContent = "Reviewed by operations lead. Ready to share.";
  });
}

function generateHandoff() {
  const dayEvents = eventsForSelectedDate();
  const lines = [
    `Handoff for ${state.selectedDate}`,
    "",
    "Today",
    ...dayEvents.map(
      (event) =>
        `- ${event.startTime} ${event.orderNumber} ${event.eventName}: ${event.readiness}. ${event.readinessSummary}`,
    ),
    "",
    "Open review",
    ...state.reviewItems
      .filter((item) => item.status === "Open")
      .map((item) => {
        const event = eventById(item.eventId);
        return `- ${item.severity}: ${event?.orderNumber || item.eventId} - ${item.title}`;
      }),
    "",
    "Incomplete tasks",
    ...state.taskItems
      .filter((item) => item.status !== "Done" && item.approvalState === "approved")
      .map((item) => {
        const event = eventById(item.eventId);
        return `- ${item.dueTime}: ${item.ownerRole} - ${item.title} (${event?.orderNumber || item.eventId})`;
      }),
    "",
    "Human review note",
    "- AI-generated handoff requires coordinator review before sharing.",
  ];
  return lines.join("\n");
}

function groupBy(items, key) {
  return items.reduce((groups, item) => {
    const group = item[key] || "Other";
    groups[group] = groups[group] || [];
    groups[group].push(item);
    return groups;
  }, {});
}

function render() {
  const path = currentPath();
  if (path === "/today") renderToday();
  else if (path === "/events") renderEvents();
  else if (path.startsWith("/events/")) renderEventDetail(path.split("/").pop());
  else if (path === "/production") renderProduction();
  else if (path === "/setup-delivery") renderSetupDelivery();
  else if (path === "/tasks") renderTasks();
  else if (path === "/review") renderReview();
  else if (path === "/handoff") renderHandoff();
  else navigate("/today");
}

document.addEventListener("click", (event) => {
  const routeTarget = event.target.closest("[data-route]");
  if (routeTarget) {
    event.preventDefault();
    navigate(routeTarget.dataset.route);
    return;
  }

  const taskToggle = event.target.closest("[data-task-toggle]");
  if (taskToggle) {
    const task = state.taskItems.find((item) => item.id === taskToggle.dataset.taskToggle);
    if (task && task.approvalState === "approved") {
      task.status = task.status === "Done" ? "Todo" : "Done";
      render();
    }
    return;
  }

  const taskApprove = event.target.closest("[data-task-approve]");
  if (taskApprove) {
    const task = state.taskItems.find((item) => item.id === taskApprove.dataset.taskApprove);
    if (task) {
      task.approvalState = "approved";
      task.status = "Todo";
      render();
    }
    return;
  }

  const accept = event.target.closest("[data-review-accept]");
  const defer = event.target.closest("[data-review-defer]");
  const dismiss = event.target.closest("[data-review-dismiss]");
  const action = accept || defer || dismiss;
  if (action) {
    const item = state.reviewItems.find(
      (review) =>
        review.id === action.dataset.reviewAccept ||
        review.id === action.dataset.reviewDefer ||
        review.id === action.dataset.reviewDismiss,
    );
    if (item) {
      if (accept) {
        item.status = "Accepted";
        item.humanDecision = "Accepted for operational follow-up.";
      }
      if (defer) {
        item.status = "Deferred";
        item.humanDecision = "Deferred by coordinator for later review.";
      }
      if (dismiss) {
        item.status = "Dismissed";
        item.humanDecision = "Dismissed by coordinator with source context visible.";
      }
      state.handoffText = "";
      render();
    }
  }
});

window.addEventListener("popstate", render);

if (window.location.pathname === "/") {
  window.history.replaceState({}, "", "/today");
}
render();
