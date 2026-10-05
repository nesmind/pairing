/**
 * Image-to-image controls of the Images page: the Mode select, the source-image picker and its strength. Exposes
 * window.generationSource for image_generation.js; the picture is resized in the browser to exactly the requested
 * width x height and sent as a base64 PNG (never stored by the server).
 */
(() => {
  const MAX_SIDE = 512;
  const modeSelect = document.getElementById("gen-mode");
  const panel = document.getElementById("gen-source-panel");
  const fileInput = document.getElementById("gen-source-file");
  const preview = document.getElementById("gen-source-preview");
  let source = null; // the decoded <img>

  modeSelect.addEventListener("change", () => panel.classList.toggle("hidden", modeSelect.value !== "image_to_image"));

  // Size inputs follow the picture (long side <= MAX_SIDE, multiples of 64 - what the model works in).
  const snap = (value) => Math.max(64, Math.round(value / 64) * 64);
  function sizeFor(img) {
    const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
    return [snap(img.naturalWidth * scale), snap(img.naturalHeight * scale)];
  }

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    source = null;
    preview.classList.add("hidden");
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      source = img;
      [document.getElementById("gen-width").value, document.getElementById("gen-height").value] = sizeFor(img);
      preview.src = url;
      preview.classList.remove("hidden");
    };
    img.onerror = () => URL.revokeObjectURL(url);
    img.src = url;
  });

  window.generationSource = {
    isImageMode: () => modeSelect.value === "image_to_image",
    hasImage: () => source !== null,
    strength: () => parseFloat(document.getElementById("gen-strength").value),
    // Base64 PNG of the picture drawn at exactly width x height.
    pngBase64(width, height) {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      canvas.getContext("2d").drawImage(source, 0, 0, width, height);
      return canvas.toDataURL("image/png").split(",")[1];
    },
  };
})();
