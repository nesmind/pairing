/* Settings tabs with a "Configure..." selector on top (Inference engines, Knowledge): it picks which `[data-group]` block of the tab shows. The first <option> is the starting choice on every load (nothing is remembered). Hidden blocks stay in the DOM, so settings.js keeps binding their sections. A new option is one <option value="x"> plus one <div data-group="x">. */
class GroupSelect {
  constructor(select) {
    this.select = select;
    this.groups = select.closest("[id^='tab-']").querySelectorAll("[data-group]");
    select.value = select.options[0].value;
    select.addEventListener("change", () => this.apply());
    this.apply();
  }

  apply() {
    for (const group of this.groups) {
      group.classList.toggle("hidden", group.dataset.group !== this.select.value);
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  for (const select of document.querySelectorAll("select[data-group-select]")) new GroupSelect(select);
});
