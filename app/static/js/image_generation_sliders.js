/**
 * Images page: the compact range sliders (Width/Height/Steps/CFG) show their value beside the label. A script that
 * sets a slider's value (e.g. the source picture sizing Width/Height) dispatches "input" to refresh it. Also the
 * seed's Random button.
 */
(() => {
  for (const slider of document.querySelectorAll("[data-slider]")) {
    const out = document.getElementById(`${slider.id}-value`);
    const show = () => (out.textContent = slider.value);
    slider.addEventListener("input", show);
    show();
  }
  document.getElementById("gen-seed-random").addEventListener("click", () => {
    document.getElementById("gen-seed").value = -1;
  });
})();
