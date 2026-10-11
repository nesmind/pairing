/** "Reset cluster": every node back to a fresh install and cluster mode off, for starting over (tests). Shown while
 * a cluster is set up. Needs no quorum. The nodes are reset several at a time and each one is listed as it
 * finishes (server-sent events from POST /api/cluster/reset/stream), so a big cluster shows its progress and which
 * nodes need the reset script run by hand. */

const RESET_OK_SHOWN = 200;  // the long list of nodes that went fine is capped; failures are always all shown

let _resetSectionHidden = true;

function renderClusterReset(state) {
  const shown = state.supported && state.enabled;
  if (shown && _resetSectionHidden) _clearResetResult();  // a new cluster is being set up: the last reset's report is old
  _resetSectionHidden = !shown;
  document.getElementById("cluster-reset").classList.toggle("hidden", !shown);
}

function _clearResetResult() {
  for (const id of ["cluster-reset-progress", "cluster-reset-note"]) {
    document.getElementById(id).classList.add("hidden");
  }
  document.getElementById("cluster-reset-failed").innerHTML = document.getElementById("cluster-reset-ok").innerHTML = "";
}

function _resetProgress(total, ok, failed) {
  const done = ok + failed;
  document.getElementById("cluster-reset-bar").style.width = `${total ? Math.round((100 * done) / total) : 100}%`;
  document.getElementById("cluster-reset-count").textContent =
    `${done} of ${total} node${total === 1 ? "" : "s"}: ${ok} reset` + (failed ? `, ${failed} need the script` : "");
  document.getElementById("cluster-reset-ok-count").textContent = ok;
}

function _resetNodeLine(event, shown) {
  const line = document.createElement("li");
  if (event.ok) {
    if (shown >= RESET_OK_SHOWN) return null;
    line.textContent = `✓ ${event.host}`;
  } else {
    line.textContent = `✗ ${event.host}: ${event.reason}. Run scripts/reset_cluster_node.sh on that machine.`;
  }
  return line;
}

async function _readResetEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      if (chunk.startsWith("data: ")) onEvent(JSON.parse(chunk.slice(6)));
    }
  }
}

async function resetWholeCluster() {
  const ok = confirm(
    "Reset the whole cluster?\n\nEvery node forgets the cluster (setup, history, node identity) and its own saved chat caches, and restarts as a fresh install; " +
      "cluster mode goes off and pAIring forgets the shared folders and cluster settings; the hosts stay as plain remote hosts. Models and shared folders are not touched. This cannot be undone."
  );
  if (!ok) return;
  const note = document.getElementById("cluster-reset-note");
  const box = document.getElementById("cluster-reset-progress");
  const failedList = document.getElementById("cluster-reset-failed");
  const okList = document.getElementById("cluster-reset-ok");
  failedList.innerHTML = okList.innerHTML = "";
  box.classList.remove("hidden");
  note.classList.remove("hidden");
  note.textContent = "Resetting…";
  note.scrollIntoView({ behavior: "smooth", block: "center" });  // the progress is below the button: show it
  let total = 0, okCount = 0, failedCount = 0;
  _resetProgress(0, 0, 0);
  try {
    const response = await fetch("/api/cluster/reset/stream", { method: "POST" });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || response.statusText);
    await _readResetEvents(response, (event) => {
      if (event.event === "start") {
        total = event.total;
      } else if (event.event === "node") {
        const line = _resetNodeLine(event, okCount);
        event.ok ? okCount++ : failedCount++;
        if (line) (event.ok ? okList : failedList).appendChild(line);
      } else if (event.event === "done") {
        note.textContent = failedCount
          ? "Done, but some nodes were not reset: run the script on each of them (listed below), then reload the page."
          : "Done: every node was reset and is restarting. Reloading…";
        if (!failedCount) setTimeout(() => location.reload(), 2500);  // the list of failures would be lost, so only then
      }
      _resetProgress(total, okCount, failedCount);
    });
  } catch (err) {
    note.textContent = err.message;
  }
}

document.getElementById("cluster-reset-btn").addEventListener("click", resetWholeCluster);
