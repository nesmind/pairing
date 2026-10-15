/* Connectors page: fetches every registered connector (app.services.connectors, app/routers/connectors.py) and renders a compact row each, with a config form built from its config_fields, so a new connector needs only a backend ConnectorDefinition. Rows start collapsed (the hidden + rotating-chevron idiom of Settings > Model's "Browse more models"); opening one collapses the other, so at most one is expanded. */

// The expanded connector's form, re-applied across a loadConnectors re-render (after a save/toggle) so the row being
// edited doesn't collapse.
let openConnectorId = null;

// Two independent columns from md up (alternating, so reading order stays left, right, left...): opening a card then
// only moves the cards below it in its own column, never its neighbour, as a shared-row grid did. One column below md.
const twoColumns = window.matchMedia("(min-width: 768px)");
let connectorCards = [];

function layoutConnectorCards() {
  const listEl = document.getElementById("connectors-list");
  const columnCount = twoColumns.matches ? 2 : 1;
  const columns = Array.from({ length: columnCount }, () => {
    const column = document.createElement("div");
    column.className = "flex min-w-0 flex-1 flex-col gap-4";
    return column;
  });
  connectorCards.forEach((card, index) => columns[index % columnCount].appendChild(card));
  listEl.replaceChildren(...columns);
}

twoColumns.addEventListener("change", layoutConnectorCards);

/* The full description of a card, floating next to the pointer after a short pause over the card's top row (the line
on the card is cut to fit). One shared box, inverted colours so it stands out in every theme, kept inside the window. */
class DescriptionPopover {
  constructor(delayMs = 500) {
    this.delayMs = delayMs;
    this.timer = null;
    this.box = document.createElement("div");
    this.box.className =
      "pointer-events-none fixed z-50 max-w-[22rem] rounded-md bg-slate-100 px-3 py-2 text-xs leading-relaxed " +
      "text-slate-900 shadow-xl opacity-0 transition-opacity duration-150";
    this.box.hidden = true;
    this.text = document.createElement("div");
    this.arrow = document.createElement("div");
    this.arrow.className = "absolute h-2.5 w-2.5 rotate-45 bg-slate-100";
    this.box.append(this.arrow, this.text);
    document.body.appendChild(this.box);
    window.addEventListener("scroll", () => this.hide(), true);
  }

  attach(target, text) {
    let pointer = { x: 0, y: 0 };
    target.addEventListener("mouseenter", (e) => {
      pointer = { x: e.clientX, y: e.clientY };
      clearTimeout(this.timer);
      this.timer = setTimeout(() => this.show(text, pointer), this.delayMs);
    });
    target.addEventListener("mousemove", (e) => {
      pointer = { x: e.clientX, y: e.clientY };
      if (!this.box.hidden) this.place(pointer);
    });
    target.addEventListener("mouseleave", () => this.hide());
    target.addEventListener("click", () => this.hide());
  }

  show(text, pointer) {
    this.text.textContent = text;
    this.box.hidden = false;
    this.place(pointer);
    requestAnimationFrame(() => this.box.classList.replace("opacity-0", "opacity-100"));
  }

  hide() {
    clearTimeout(this.timer);
    this.box.classList.replace("opacity-100", "opacity-0");
    this.box.hidden = true;
  }

  /* Below and right of the pointer; above it when there is no room below; shifted left near the window's edge. The
  arrow points back at the pointer. */
  place({ x, y }) {
    const gap = 14;
    const margin = 8;
    const { width, height } = this.box.getBoundingClientRect();
    const left = Math.max(margin, Math.min(x - 16, window.innerWidth - width - margin));
    const below = y + gap + height + margin <= window.innerHeight;
    const top = below ? y + gap : y - gap - height;
    this.box.style.left = `${left}px`;
    this.box.style.top = `${top}px`;
    this.arrow.style.left = `${Math.max(8, Math.min(x - left - 5, width - 18))}px`;
    this.arrow.style.top = below ? "-5px" : "";
    this.arrow.style.bottom = below ? "" : "-5px";
  }
}

const descriptionPopover = new DescriptionPopover();

async function loadConnectors() {
  const errorEl = document.getElementById("connectors-error");
  try {
    const { connectors } = await api("/api/connectors");
    connectorCards = connectors.map((connector) => renderConnectorCard(connector));
    layoutConnectorCards();
    errorEl.classList.add("hidden");
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.classList.remove("hidden");
  }
}

function renderConnectorCard(connector) {
  const card = document.getElementById("connector-card-template").content.firstElementChild.cloneNode(true);

  card.querySelector(".connector-name").textContent = connector.display_name;
  card.querySelector(".connector-name").title = connector.display_name;
  card.querySelector(".connector-description").textContent = connector.description;
  descriptionPopover.attach(card.querySelector(".connector-card-head"), connector.description);
  card.querySelector(".connector-ready-badge").classList.toggle("hidden", !connector.ready);

  const detailEl = card.querySelector(".connector-detail");
  const chevronEl = card.querySelector(".connector-chevron");
  const isOpen = openConnectorId === connector.id;
  detailEl.classList.toggle("hidden", !isOpen);
  chevronEl.classList.toggle("rotate-180", isOpen);

  card.querySelector(".connector-card-toggle").addEventListener("click", () => {
    const opening = detailEl.classList.contains("hidden");
    // Accordion: collapse the other forms first.
    document.querySelectorAll("#connectors-list .connector-detail").forEach((el) => el.classList.add("hidden"));
    document.querySelectorAll("#connectors-list .connector-chevron").forEach((el) => el.classList.remove("rotate-180"));
    detailEl.classList.toggle("hidden", !opening);
    chevronEl.classList.toggle("rotate-180", opening);
    openConnectorId = opening ? connector.id : null;
  });

  const enabledToggle = card.querySelector(".connector-enabled-toggle");
  enabledToggle.checked = connector.enabled;
  const enabledStatus = card.querySelector(".connector-enabled-status");
  enabledToggle.addEventListener("change", async () => {
    const desired = enabledToggle.checked;
    try {
      await api(`/api/connectors/${connector.id}/enabled`, {
        method: "PUT",
        body: JSON.stringify({ enabled: desired }),
      });
      await loadConnectors();
    } catch (err) {
      enabledToggle.checked = !desired;
      enabledStatus.textContent = err.message;
    }
  });

  const fieldsEl = card.querySelector(".connector-fields");
  const fieldTemplate = document.getElementById("connector-field-template");
  const inputsByName = {};
  connector.config_fields.forEach((field) => {
    const row = fieldTemplate.content.firstElementChild.cloneNode(true);
    const input = row.querySelector(".connector-field-input");
    row.querySelector(".connector-field-label").textContent = field.label + (field.required ? " *" : "");
    input.type = field.field_type === "password" ? "password" : "text";
    if (field.secret) {
      // Secrets are never re-filled (connector_config_service.get_config_for_display); only whether one is saved. A
      // blank submission keeps the existing value.
      input.placeholder = connector.values[`has_${field.name}`] ? "•••••••• (saved — leave blank to keep)" : "";
    } else {
      input.value = connector.values[field.name] || "";
    }
    row.querySelector(".connector-field-help").textContent = field.help_text;
    inputsByName[field.name] = input;
    fieldsEl.appendChild(row);
  });

  const saveStatus = card.querySelector(".connector-save-status");
  const testBtn = card.querySelector(".connector-test-btn");
  const resultEl = card.querySelector(".connector-test-result");
  const formValues = () => Object.fromEntries(Object.entries(inputsByName).map(([name, input]) => [name, input.value]));

  const showResult = (passed, message) => {
    resultEl.textContent = `${passed ? "✓" : "✗"} ${message}`;
    resultEl.classList.remove("hidden", "text-emerald-400", "text-red-400");
    resultEl.classList.add(passed ? "text-emerald-400" : "text-red-400");
  };

  // Tests the form's values, saved or not; a test of the saved values is recorded and may make it ready.
  const runTest = async (values) => {
    testBtn.disabled = true;
    saveStatus.textContent = "Testing…";
    try {
      const result = await api(`/api/connectors/${connector.id}/test-values`, {
        method: "POST",
        body: JSON.stringify({ values }),
      });
      if (result.saved) {
        await loadConnectors();
        return;
      }
      saveStatus.textContent = "";
      showResult(result.passed, result.passed ? `${result.message} Not saved yet — press Save to keep these values.` : result.message);
    } catch (err) {
      saveStatus.textContent = err.message;
    } finally {
      testBtn.disabled = false;
    }
  };

  testBtn.addEventListener("click", () => runTest(formValues()));

  card.querySelector(".connector-config-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    saveStatus.textContent = "Saving…";
    try {
      await api(`/api/connectors/${connector.id}/config`, {
        method: "PUT",
        body: JSON.stringify({ values: formValues() }),
      });
      // Saving clears the last test, so test the saved values at once.
      await runTest(formValues());
    } catch (err) {
      saveStatus.textContent = err.message;
    }
  });

  if (connector.last_test_message) showResult(connector.last_test_passed, connector.last_test_message);

  return card;
}

loadConnectors();
