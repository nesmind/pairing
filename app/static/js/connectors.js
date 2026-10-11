/* Connectors page: fetches every registered connector (app.services.connectors, app/routers/connectors.py) and renders a compact row each, with a config form built from its config_fields, so a new connector needs only a backend ConnectorDefinition. Rows start collapsed (the hidden + rotating-chevron idiom of Settings > Model's "Browse more models"); opening one collapses the other, so at most one is expanded. */

// The expanded connector's form, re-applied across a loadConnectors re-render (after a save/toggle) so the row being
// edited doesn't collapse.
let openConnectorId = null;

async function loadConnectors() {
  const errorEl = document.getElementById("connectors-error");
  const listEl = document.getElementById("connectors-list");
  try {
    const { connectors } = await api("/api/connectors");
    listEl.innerHTML = "";
    connectors.forEach((connector) => listEl.appendChild(renderConnectorCard(connector)));
    errorEl.classList.add("hidden");
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.classList.remove("hidden");
  }
}

function renderConnectorCard(connector) {
  const card = document.getElementById("connector-card-template").content.firstElementChild.cloneNode(true);

  card.querySelector(".connector-name").textContent = connector.display_name;
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
  card.querySelector(".connector-config-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const values = {};
    Object.entries(inputsByName).forEach(([name, input]) => {
      values[name] = input.value;
    });
    try {
      await api(`/api/connectors/${connector.id}/config`, {
        method: "PUT",
        body: JSON.stringify({ values }),
      });
      await loadConnectors();
    } catch (err) {
      saveStatus.textContent = err.message;
    }
  });

  const testBtn = card.querySelector(".connector-test-btn");
  const testStatus = card.querySelector(".connector-test-status");
  testBtn.disabled = !connector.configured;
  testBtn.title = connector.configured ? "" : "Save the required fields above first";
  testBtn.addEventListener("click", async () => {
    testBtn.disabled = true;
    testStatus.textContent = "Testing…";
    try {
      await api(`/api/connectors/${connector.id}/test`, { method: "POST" });
      await loadConnectors();
    } catch (err) {
      testStatus.textContent = err.message;
      testBtn.disabled = false;
    }
  });

  const resultEl = card.querySelector(".connector-test-result");
  if (connector.last_test_message) {
    resultEl.textContent = connector.last_test_passed ? `✓ ${connector.last_test_message}` : `✗ ${connector.last_test_message}`;
    resultEl.classList.remove("hidden", "text-emerald-400", "text-red-400");
    resultEl.classList.add(connector.last_test_passed ? "text-emerald-400" : "text-red-400");
  }

  return card;
}

loadConnectors();
