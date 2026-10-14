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
