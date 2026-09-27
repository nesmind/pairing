/**
 * Connectors page — fetches every registered connector (see
 * app.services.connectors, app/routers/connectors.py) and renders one compact row per
 * connector, its config form built generically from that connector's own
 * config_fields so a new connector needs no change here, only a new
 * ConnectorDefinition on the backend. Each row is collapsed by default (same
 * hidden + rotating-chevron idiom as Settings > Model's "Browse more models" panel)
 * — clicking it reveals the form, and opening one collapses whichever other
 * connector's form was open, so at most one is ever expanded at a time.
 */

// Which connector's form is currently expanded, if any — re-applied across a
// loadConnectors() re-render (after a save/enable toggle) so the row the admin was
// just working in doesn't collapse out from under them.
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
    // Accordion: collapse every other connector's form first, so at most one is ever open.
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
      // Never re-filled with the real value (see connector_config_service.get_config_for_display) — only
      // whether one is already saved. A blank submission for a secret field means "keep the existing value".
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
