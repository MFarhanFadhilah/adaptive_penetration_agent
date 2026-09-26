let selectedChallenge = null;
let evtSource = null;
let series = { naive: [], memory: [] };
let runPasswordRequired = false;
let runPassword = "";

function fmtTime(ts) {
  return new Date(ts * 1000).toLocaleTimeString();
}

async function refreshStatus() {
  try {
    const r = await fetch("/api/status");
    const s = await r.json();
    const atlasBadge = document.getElementById("atlas-status");
    const gwBadge = document.getElementById("gateway-status");
    if (s.connected) {
      atlasBadge.textContent = `Atlas: connected (${Object.entries(s.collections).map(([k, v]) => `${k}=${v}`).join(", ")})`;
      atlasBadge.className = "badge badge-ok";
    } else {
      atlasBadge.textContent = "Atlas: disconnected";
      atlasBadge.className = "badge badge-bad";
    }
    runPasswordRequired = !!s.run_password_required;
    gwBadge.textContent = s.ai_gateway_configured ? "AI Gateway: configured (best-effort)" : "AI Gateway: not configured (scripted reasoning)";
    gwBadge.className = "badge " + (s.ai_gateway_configured ? "badge-ok" : "badge-pending");
  } catch (e) {
    document.getElementById("atlas-status").textContent = "Atlas: error";
    document.getElementById("atlas-status").className = "badge badge-bad";
  }
}

async function refreshMemory() {
  const r = await fetch("/api/memory");
  const data = await r.json();

  const policyTable = document.getElementById("policy-table");
  policyTable.innerHTML = "";
  const p = data.policy || {};
  for (const k of ["version", "token_budget", "rag_top_k", "rag_inject_every", "truncate_content_chars", "alpha", "beta"]) {
    if (!(k in p)) continue;
    const row = document.createElement("tr");
    row.innerHTML = `<td class="muted">${k}</td><td>${p[k]}</td>`;
    policyTable.appendChild(row);
  }

  renderMemory(data.semantic_memory, false);
  return data.semantic_memory;
}

// Memory table state, used to show what the last run changed.
let currentMemory = [];      // docs as last rendered, in rank order
let memorySnapshot = null;   // copy of currentMemory taken when a run starts
let lastRunUsed = new Set(); // doc_ids injected as hints in the last run
let lastDeltas = {};         // doc_id -> {dUtility, rankFrom, rankTo}

function renderMemory(docs, flash) {
  currentMemory = docs;
  const rows = document.getElementById("memory-rows");
  rows.innerHTML = "";
  docs.forEach((d) => {
    const delta = lastDeltas[d.doc_id];
    const used = lastRunUsed.has(d.doc_id);
    const tr = document.createElement("tr");
    if (used) tr.classList.add("row-used");
    if (flash && delta && delta.dUtility !== 0) tr.classList.add(delta.dUtility > 0 ? "flash-up" : "flash-down");

    let change = "";
    if (delta && delta.dUtility !== 0) {
      const up = delta.dUtility > 0;
      change = `<span class="${up ? "delta-up" : "delta-down"}">${up ? "▲ +" : "▼ "}${delta.dUtility.toFixed(3)}</span>`;
      const moved = delta.rankFrom - delta.rankTo;
      if (moved !== 0) change += ` <span class="muted">rank #${delta.rankFrom + 1}→#${delta.rankTo + 1}</span>`;
    }
    tr.innerHTML = `<td>${d.doc_id}${used ? ' <span class="tag-used">used</span>' : ""}</td>` +
      `<td>${d.category}</td><td>${d.title}</td>` +
      `<td>${d.times_retrieved}</td><td>${d.times_led_to_success}</td>` +
      `<td>${d.utility_score.toFixed(3)}</td><td>${change}</td>`;
    rows.appendChild(tr);
  });
}

// The Atlas Trigger on `runs` updates semantic_memory asynchronously, so poll
// until every doc used this run shows the extra retrieval (or give up).
async function waitForTriggerUpdate(usedDocIds, snapshot, timeoutMs = 10000) {
  const before = Object.fromEntries(snapshot.map((d) => [d.doc_id, d.times_retrieved]));
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const data = await (await fetch("/api/memory")).json();
    const docs = data.semantic_memory;
    const updated = usedDocIds.every((id) => {
      const doc = docs.find((d) => d.doc_id === id);
      return doc && doc.times_retrieved > (before[id] ?? 0);
    });
    if (updated) return { docs, seconds: (Date.now() - started) / 1000 };
    await new Promise((r) => setTimeout(r, 700));
  }
  return { docs: null, seconds: null };
}

async function showLearning(usedDocIds) {
  lastRunUsed = new Set(usedDocIds);
  if (!memorySnapshot || usedDocIds.length === 0) {
    lastDeltas = {};
    await refreshMemory();
    return;
  }
  const snapshot = memorySnapshot;
  memorySnapshot = null;
  const { docs, seconds } = await waitForTriggerUpdate(usedDocIds, snapshot);
  if (!docs) {
    log(`<span style="color:#f56565">Atlas Trigger did not update semantic_memory within 10s — check that the trigger watches the runs collection.</span>`);
    lastDeltas = {};
    await refreshMemory();
    return;
  }

  const beforeRank = Object.fromEntries(snapshot.map((d, i) => [d.doc_id, { rank: i, utility: d.utility_score }]));
  lastDeltas = {};
  const changes = [];
  docs.forEach((d, i) => {
    const b = beforeRank[d.doc_id];
    if (!b) return;
    const dUtility = d.utility_score - b.utility;
    if (Math.abs(dUtility) < 1e-9 && b.rank === i) return;
    lastDeltas[d.doc_id] = { dUtility, rankFrom: b.rank, rankTo: i };
    if (Math.abs(dUtility) >= 1e-9) {
      changes.push(`${d.doc_id} ${b.utility.toFixed(3)}→${d.utility_score.toFixed(3)}`);
    }
  });
  renderMemory(docs, true);
  log(`<b>Memory learned from this run</b> — Atlas Trigger updated ${changes.length} doc(s) in ${seconds.toFixed(1)}s: ${changes.join(", ")}`, "log-learn");
}

async function refreshRuns() {
  const r = await fetch("/api/runs?simulated=0&limit=25");
  const runs = await r.json();
  const rows = document.getElementById("runs-rows");
  rows.innerHTML = "";
  for (const run of runs) {
    const tr = document.createElement("tr");
    const kind = run.simulated ? '<span class="muted">sim</span>' : '<span class="tag-real">real</span>';
    tr.innerHTML = `<td>${run.challenge} ${kind}</td><td>${run.category}</td>` +
      `<td class="${run.solved ? "solved-yes" : "solved-no"}">${run.solved ? "yes" : "no"}</td>` +
      `<td>${run.tokens_used}</td><td>$${run.cost}</td><td>${fmtTime(run.timestamp)}</td>`;
    rows.appendChild(tr);
  }
}

async function loadChallenges() {
  const r = await fetch("/api/challenges");
  const challenges = await r.json();
  const grid = document.getElementById("challenge-grid");
  grid.innerHTML = "";
  for (const c of challenges) {
    const el = document.createElement("div");
    el.className = "challenge-card";
    el.dataset.id = c.id;
    el.innerHTML = `<div class="cat">${c.category}${c.points ? " · " + c.points + "pt" : ""}</div>` +
      `<div class="name">${c.name}</div><div class="meta">${c.event} ${c.year}</div>`;
    el.addEventListener("click", () => selectChallenge(c, el));
    grid.appendChild(el);
  }
}

function selectChallenge(c, el) {
  selectedChallenge = c;
  document.querySelectorAll(".challenge-card").forEach(x => x.classList.remove("selected"));
  el.classList.add("selected");
  setRunButtons(false);
}

function setRunButtons(running) {
  const real = document.getElementById("real-btn");
  const sim = document.getElementById("run-btn");
  real.disabled = running || !selectedChallenge;
  sim.disabled = running || !selectedChallenge;
  if (!selectedChallenge) return;
  real.textContent = running ? "Running…" : `Run real agent: ${selectedChallenge.name}`;
  sim.textContent = running ? "Running…" : "Simulate";
}

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// Highlight the harness lines that matter for the MongoDB story
function realLineClass(line) {
  if (line.startsWith("[Mongo-RAG]")) return "log-hint";
  if (line.startsWith("[Policy]") || line.startsWith("[RAG Mode")) return "log-learn";
  return "";
}

function runReal() {
  if (!selectedChallenge || evtSource) return;
  const maxCost = document.getElementById("max-cost").value || "1.0";
  if (!confirm(`Run the real agent on ${selectedChallenge.name}?

This starts Docker containers and uses your OpenAI API key (budget: $${maxCost}). It can take several minutes.`)) return;
  if (runPasswordRequired && !runPassword) {
    runPassword = prompt("Run password (set by the server owner):") || "";
    if (!runPassword) return;
  }
  document.getElementById("log").innerHTML = "";
  document.getElementById("summary-card").hidden = true;
  resetChart();
  memorySnapshot = currentMemory.map((d) => ({ ...d }));
  setRunButtons(true);

  evtSource = new EventSource(`/api/run/stream?challenge=${encodeURIComponent(selectedChallenge.id)}` +
    `&max_cost=${encodeURIComponent(maxCost)}&password=${encodeURIComponent(runPassword)}`);
  let finished = false;

  const finish = () => {
    finished = true;
    if (evtSource) { evtSource.close(); evtSource = null; }
    setRunButtons(false);
  };

  evtSource.addEventListener("start", (e) => {
    const d = JSON.parse(e.data);
    log(`<b>Real agent started</b> — <code>${escapeHtml(d.command)}</code> (budget $${d.max_cost})`);
  });

  evtSource.addEventListener("log", (e) => {
    const d = JSON.parse(e.data);
    log(`<span class="log-raw">${escapeHtml(d.line)}</span>`, realLineClass(d.line));
  });

  evtSource.addEventListener("failed", (e) => {
    const d = JSON.parse(e.data);
    log(`<span style="color:#f56565"><b>Cannot run:</b> ${escapeHtml(d.message)}</span>`);
    if (d.message === "Wrong run password.") runPassword = "";
    finish();
  });

  evtSource.addEventListener("done", (e) => {
    const d = JSON.parse(e.data);
    const run = d.run;
    finish();
    const card = document.getElementById("summary-card");
    card.hidden = false;
    if (!run) {
      document.getElementById("summary").innerHTML =
        `<div>Agent exited with code ${d.exit_code} before writing a result to Atlas — see the log above.</div>`;
      return;
    }
    document.getElementById("summary").innerHTML = `
      <div>Solved: <span class="${run.solved ? "solved-yes" : "solved-no"}">${run.solved ? "YES" : "NO"}</span> <span class="tag-real">real agent run</span></div>
      <div>Tokens: <b>${run.tokens_used}</b> (input ${run.input_tokens}, output ${run.output_tokens})</div>
      <div>Cost: $${Number(run.cost).toFixed(4)}</div>
      <div>Memory docs used (real Atlas doc_ids): ${run.used_doc_ids.length ? run.used_doc_ids.join(", ") : "none"}</div>
      <div>Policy version at start: v${run.policy_version}</div>
    `;
    log(`<b>Episode finished — written to Atlas runs collection, outcome trigger fired for: ${run.used_doc_ids.join(", ") || "none"}</b>`);
    showLearning(run.used_doc_ids);
    refreshRuns();
    refreshMemory();
  });

  evtSource.onerror = () => {
    if (finished) return;
    log(`<span style="color:#f56565">Stream closed unexpectedly — the agent was stopped.</span>`);
    finish();
  };
}

function log(html, cls) {
  const box = document.getElementById("log");
  const line = document.createElement("div");
  line.className = "log-line" + (cls ? " " + cls : "");
  line.innerHTML = html;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

function resetChart() {
  series = { naive: [], memory: [] };
  drawChart();
}

function drawChart() {
  const canvas = document.getElementById("chart");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, pad = 30;
  ctx.clearRect(0, 0, W, H);

  const all = series.naive.concat(series.memory);
  const maxY = Math.max(1000, ...all);
  const n = Math.max(series.naive.length, series.memory.length, 1);

  ctx.strokeStyle = "#2a313c";
  ctx.beginPath();
  ctx.moveTo(pad, H - pad); ctx.lineTo(W - 10, H - pad);
  ctx.moveTo(pad, H - pad); ctx.lineTo(pad, 10);
  ctx.stroke();

  ctx.fillStyle = "#8a93a1";
  ctx.font = "10px monospace";
  ctx.fillText(Math.round(maxY) + " tok", 4, 16);
  ctx.fillText("0", 14, H - pad + 12);

  function plot(arr, color) {
    if (arr.length === 0) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    arr.forEach((v, i) => {
      const x = pad + (i / Math.max(1, n - 1)) * (W - pad - 20);
      const y = H - pad - (v / maxY) * (H - pad - 20);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  plot(series.naive, "#f56565");
  plot(series.memory, "#4fd1c5");
}

function runSimulation() {
  if (!selectedChallenge || evtSource) return;
  document.getElementById("log").innerHTML = "";
  document.getElementById("summary-card").hidden = true;
  resetChart();
  memorySnapshot = currentMemory.map((d) => ({ ...d }));

  setRunButtons(true);

  evtSource = new EventSource(`/api/simulate/stream?challenge=${encodeURIComponent(selectedChallenge.id)}`);

  evtSource.addEventListener("start", (e) => {
    const d = JSON.parse(e.data);
    log(`<b>Episode ${d.episode_id}</b> — policy: top_k=${d.policy.rag_top_k}, inject_every=${d.policy.rag_inject_every}, truncate=${d.policy.truncate_content_chars}c`);
  });

  evtSource.addEventListener("hint", (e) => {
    const d = JSON.parse(e.data);
    d.hints.forEach(h => {
      log(`<span class="log-hint">[Mongo-RAG hint round ${d.round}] ${h.doc_title} (sim=${h.vscore.toFixed(2)}, utility=${h.utility_score.toFixed(2)})</span> — real Atlas Vector Search result`, "log-hint");
    });
  });

  evtSource.addEventListener("warning", (e) => {
    const d = JSON.parse(e.data);
    log(`<span style="color:#f56565">[warning round ${d.round}] ${d.message}</span>`);
  });

  evtSource.addEventListener("round", (e) => {
    const d = JSON.parse(e.data);
    series.naive.push(d.naive_context_tokens);
    series.memory.push(d.memory_context_tokens);
    drawChart();
    log(`<span class="log-round">Round ${d.round}</span> — <code>${d.tool}</code>` +
      `<span class="log-tag">${d.reasoning_source}</span><br/>` +
      `<i>${d.thought}</i><br/>` +
      `<span class="muted">output (${d.full_output_chars} chars, truncated in-context): ${d.output_preview.slice(0, 120)}...</span><br/>` +
      `<span class="muted">context this round — without memory: ${d.naive_context_tokens} tok, with memory: ${d.memory_context_tokens} tok</span>`);
  });

  evtSource.addEventListener("done", (e) => {
    const d = JSON.parse(e.data);
    const card = document.getElementById("summary-card");
    card.hidden = false;
    document.getElementById("summary").innerHTML = `
      <div>Solved: <span class="${d.solved ? "solved-yes" : "solved-no"}">${d.solved ? "YES" : "NO"}</span></div>
      <div>Total tokens sent across all rounds — with memory: <b>${d.tokens_used}</b>, without memory: <b>${d.naive_tokens_used}</b> (${(d.naive_tokens_used / Math.max(1, d.tokens_used)).toFixed(1)}x more without memory)</div>
      <div>Context size in the final round — with memory: <b>${d.final_memory_context}</b> tok, without memory: <b>${d.final_naive_context}</b> tok (${(d.final_naive_context / Math.max(1, d.final_memory_context)).toFixed(1)}x larger without memory)</div>
      <div>Cost: $${d.cost}</div>
      <div>Memory docs used (real Atlas doc_ids): ${d.used_doc_ids.length ? d.used_doc_ids.join(", ") : "none"}</div>
      <div>Policy self-tuning: ${d.policy_change ? JSON.stringify(d.policy_change) : "no change this run"}</div>
    `;
    log(`<b>Episode finished — written to Atlas runs collection (simulated=true), outcome trigger fired for: ${d.used_doc_ids.join(", ") || "none"}</b>`);
    evtSource.close();
    evtSource = null;
    setRunButtons(false);
    showLearning(d.used_doc_ids);
    refreshRuns();
  });

  evtSource.onerror = () => {
    log(`<span style="color:#f56565">Stream error / closed.</span>`);
    if (evtSource) { evtSource.close(); evtSource = null; }
    setRunButtons(false);
  };
}

async function resetMemory() {
  if (evtSource) {
    alert("Wait for the current run to finish before resetting.");
    return;
  }
  if (!confirm("Reset memory?\n\n• utility scores back to 0.5\n• simulated runs and episodic logs deleted\n• policy back to defaults\n\nThe 18 knowledge documents are kept.")) return;

  const btn = document.getElementById("reset-btn");
  const msg = document.getElementById("reset-msg");
  btn.disabled = true;
  btn.textContent = "Resetting…";
  try {
    const r = await fetch("/api/reset", { method: "POST" });
    const d = await r.json();
    msg.textContent = `Reset done: ${d.memory_docs_reset} memory docs reset, ${d.runs_deleted} runs and ${d.episodic_logs_deleted} episodic logs deleted, policy back to v1.`;
  } catch (e) {
    msg.textContent = "Reset failed: " + e;
  }
  msg.hidden = false;
  btn.disabled = false;
  btn.textContent = "Reset memory";
  document.getElementById("log").innerHTML = "";
  document.getElementById("summary-card").hidden = true;
  lastDeltas = {};
  lastRunUsed = new Set();
  memorySnapshot = null;
  resetChart();
  refreshStatus();
  refreshMemory();
  refreshRuns();
}

document.getElementById("run-btn").addEventListener("click", runSimulation);
document.getElementById("real-btn").addEventListener("click", runReal);
document.getElementById("reset-btn").addEventListener("click", resetMemory);

refreshStatus();
loadChallenges();
refreshMemory();
refreshRuns();
resetChart();
setInterval(refreshStatus, 8000);
