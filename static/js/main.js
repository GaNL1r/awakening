const LOG_POLL_INTERVAL_MS = 500;

function toggleFullscreen() {
  const container = document.querySelector(".video-container");
  if (!container) return;

  if (!document.fullscreenElement) {
    container
      .requestFullscreen()
      .then(() => document.body.classList.add("fullscreen-mode"))
      .catch((error) => console.warn(`fullscreen failed: ${error.message}`));
    return;
  }

  document
    .exitFullscreen()
    .then(() => document.body.classList.remove("fullscreen-mode"))
    .catch((error) => console.warn(`exit fullscreen failed: ${error.message}`));
}

function bindControls() {
  document.getElementById("fullscreenBtn")?.addEventListener("click", toggleFullscreen);
  document.getElementById("multiLineChart")?.addEventListener("change", updateCharts);
  document.getElementById("chartSelectControls")?.addEventListener("change", (event) => {
    if (event.target.matches("input[type='checkbox']")) updateCharts();
  });
  document.getElementById("applyMainRange")?.addEventListener("click", updateMainRange);

  document.addEventListener("fullscreenchange", () => {
    document.body.classList.toggle("fullscreen-mode", Boolean(document.fullscreenElement));
  });
}

document.addEventListener("DOMContentLoaded", () => {
  bindControls();
  initCharts();
  startDataLoop();
  fetchAndDisplayJsonWithTree("json-log", "/log");
  setInterval(() => fetchAndDisplayJsonWithTree("json-log", "/log"), LOG_POLL_INTERVAL_MS);
});
