"use strict";

const $ = (sel) => document.querySelector(sel);
const LABEL = (l) => l.replace(/_/g, " ");
const pct = (p) => `${(p * 100).toFixed(1)}%`;

const TRIAGE_COPY = {
  urgent: { icon: "!", title: "Prioritise for immediate read", text: "One or more time-critical findings exceed their operating threshold." },
  review: { icon: "?", title: "Abnormal: routine radiologist review", text: "Non-urgent findings detected above threshold." },
  routine: { icon: "✓", title: "No finding above threshold", text: "Study can stay in the routine worklist. A radiologist still reads every image." },
};

const state = { result: null, selected: null, image: null, card: null, verdict: null, corrected: new Set() };

/* ---------------- API ---------------- */
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* not json */ }
    throw new Error(detail);
  }
  return res.json();
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), 3500);
}

/* ---------------- tabs ---------------- */
document.querySelectorAll(".tab").forEach((btn) =>
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${btn.dataset.view}`));
    if (btn.dataset.view === "monitor") loadMonitoring();
    if (btn.dataset.view === "model") renderModelCard();
  })
);

/* ---------------- health ---------------- */
async function checkHealth() {
  const el = $("#status");
  try {
    const h = await api("/health");
    el.className = "status ok";
    $("#status-text").textContent = `Model ${h.model_version} online`;
    $("#footer-version").textContent = `model ${h.model_version}`;
    if (!state.card) state.card = await api("/api/model");
  } catch (e) {
    el.className = "status err";
    $("#status-text").textContent = "API unreachable";
  }
}

/* ---------------- upload ---------------- */
const dz = $("#dropzone");
const input = $("#file-input");
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => e.dataTransfer.files[0] && analyze(e.dataTransfer.files[0]));
input.addEventListener("change", () => input.files[0] && analyze(input.files[0]));
$("#reset-btn").addEventListener("click", reset);

function reset() {
  state.result = null;
  input.value = "";
  $("#stage").hidden = true;
  $("#viewer-tools").hidden = true;
  dz.hidden = false;
  $("#findings-card").hidden = true;
  $("#feedback-card").hidden = true;
  $("#triage-card").innerHTML = document.querySelector("#triage-card").dataset.placeholder;
}

async function analyze(file) {
  if (!/image\/(png|jpeg)/.test(file.type)) return toast("Please upload a PNG or JPEG image");
  dz.hidden = true;
  const stage = $("#stage");
  stage.hidden = false;
  stage.classList.add("loading");
  $("#heat-caption").textContent = "";
  drawImageFromFile(file);

  const form = new FormData();
  form.append("file", file);
  try {
    const result = await api("/api/predict", { method: "POST", body: form });
    state.result = result;
    state.image = await loadImage(result.image);
    state.selected = result.findings[0].label;
    renderResult();
  } catch (e) {
    toast(`Analysis failed: ${e.message}`);
    reset();
  } finally {
    stage.classList.remove("loading");
  }
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

async function drawImageFromFile(file) {
  const url = URL.createObjectURL(file);
  const img = await loadImage(url);
  const c = $("#canvas"), ctx = c.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, c.width, c.height);
  const s = Math.min(c.width / img.width, c.height / img.height);
  ctx.drawImage(img, (c.width - img.width * s) / 2, (c.height - img.height * s) / 2, img.width * s, img.height * s);
  URL.revokeObjectURL(url);
}

/* ---------------- heatmap ---------------- */
// Polynomial approximation of Google's Turbo colormap
function turbo(t) {
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return [r, g, b].map((v) => Math.round(255 * Math.min(1, Math.max(0, v))));
}

function heatCanvas(grid, size = 160) {
  const h = grid.length, w = grid[0].length;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(size, size);
  for (let y = 0; y < size; y++) {
    // bilinear interpolation of the coarse CAM grid (pixel centres aligned)
    const gy = Math.min(Math.max((y + 0.5) * h / size - 0.5, 0), h - 1);
    const y0 = Math.floor(gy), y1 = Math.min(y0 + 1, h - 1), fy = gy - y0;
    for (let x = 0; x < size; x++) {
      const gx = Math.min(Math.max((x + 0.5) * w / size - 0.5, 0), w - 1);
      const x0 = Math.floor(gx), x1 = Math.min(x0 + 1, w - 1), fx = gx - x0;
      const v = (grid[y0][x0] * (1 - fx) + grid[y0][x1] * fx) * (1 - fy) + (grid[y1][x0] * (1 - fx) + grid[y1][x1] * fx) * fy;
      const [r, g, b] = turbo(v);
      const a = Math.min(1, Math.max(0, (v - 0.15) / 0.45)); // fade out low attention
      const i = (y * size + x) * 4;
      img.data[i] = r; img.data[i + 1] = g; img.data[i + 2] = b; img.data[i + 3] = Math.round(a * 255);
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

function drawStage() {
  if (!state.image || !state.result) return;
  const c = $("#canvas"), ctx = c.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.globalAlpha = 1;
  ctx.drawImage(state.image, 0, 0, c.width, c.height);
  const show = $("#heat-toggle").checked;
  const f = state.result.findings.find((x) => x.label === state.selected);
  if (show && state.selected) {
    ctx.globalAlpha = $("#heat-opacity").value / 100;
    ctx.drawImage(heatCanvas(state.result.heatmaps[state.selected]), 0, 0, c.width, c.height);
    ctx.globalAlpha = 1;
    $("#heat-caption").textContent = `Attention for ${LABEL(state.selected)} · ${pct(f.probability)}`;
  } else {
    $("#heat-caption").textContent = "";
  }
}
$("#heat-toggle").addEventListener("change", drawStage);
$("#heat-opacity").addEventListener("input", drawStage);

/* ---------------- results ---------------- */
function renderResult() {
  const r = state.result;
  const copy = TRIAGE_COPY[r.triage];
  const flagged = r.findings.filter((f) => f.positive).map((f) => LABEL(f.label));
  $("#triage-card").innerHTML = `
    <div class="triage ${r.triage}">
      <div class="triage-badge">${copy.icon}</div>
      <div>
        <div class="triage-level">${r.triage}</div>
        <div class="triage-title">${copy.title}</div>
        <div class="muted small">${flagged.length ? "Flagged: " + flagged.join(", ") : copy.text}</div>
      </div>
    </div>
    <div class="triage-meta">
      <span>Latency <b>${r.latency_ms} ms</b></span>
      <span>Model <b>${r.model_version}</b></span>
      <span>Study ID <b>${r.prediction_id}</b></span>
    </div>`;

  const list = $("#findings");
  list.innerHTML = "";
  r.findings.forEach((f) => {
    const li = document.createElement("li");
    li.className = `finding${f.positive ? " positive" : ""}${f.urgent ? " is-urgent" : ""}${f.label === state.selected ? " selected" : ""}`;
    li.innerHTML = `
      <div class="f-name"><span>${LABEL(f.label)}</span>
        ${f.positive ? `<span class="tag ${f.urgent ? "urgent" : "flag"}">${f.urgent ? "urgent" : "flagged"}</span>` : ""}
      </div>
      <div class="f-pct">${pct(f.probability)}</div>
      <div class="f-bar"><div class="f-fill" style="width:0"></div><div class="f-th" style="left:${Math.min(f.threshold, 1) * 100}%"></div></div>`;
    li.addEventListener("click", () => {
      state.selected = f.label;
      list.querySelectorAll(".finding").forEach((n) => n.classList.remove("selected"));
      li.classList.add("selected");
      drawStage();
    });
    list.appendChild(li);
    requestAnimationFrame(() => (li.querySelector(".f-fill").style.width = `${f.probability * 100}%`));
  });

  $("#findings-card").hidden = false;
  $("#viewer-tools").hidden = false;
  renderFeedback();
  drawStage();
}

/* ---------------- feedback ---------------- */
function renderFeedback() {
  state.verdict = null;
  state.corrected = new Set(state.result.flagged);
  $("#feedback-card").hidden = false;
  $("#fb-form").hidden = true;
  $("#fb-done").hidden = true;
  document.querySelectorAll(".fb-buttons .btn").forEach((b) => b.classList.remove("selected"));
  const chips = $("#fb-labels");
  chips.innerHTML = "";
  state.result.findings.map((f) => f.label).sort().forEach((label) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = `chip${state.corrected.has(label) ? " on" : ""}`;
    b.textContent = LABEL(label);
    b.addEventListener("click", () => {
      state.corrected.has(label) ? state.corrected.delete(label) : state.corrected.add(label);
      b.classList.toggle("on");
    });
    chips.appendChild(b);
  });
}

document.querySelectorAll(".fb-buttons .btn").forEach((btn) =>
  btn.addEventListener("click", async () => {
    document.querySelectorAll(".fb-buttons .btn").forEach((b) => b.classList.toggle("selected", b === btn));
    state.verdict = btn.dataset.verdict;
    if (state.verdict === "disagree") {
      $("#fb-form").hidden = false;
    } else {
      $("#fb-form").hidden = true;
      await sendFeedback();
    }
  })
);
$("#fb-form").addEventListener("submit", (e) => { e.preventDefault(); sendFeedback(); });

async function sendFeedback() {
  try {
    await api("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prediction_id: state.result.prediction_id,
        verdict: state.verdict,
        corrected_labels: state.verdict === "disagree" ? [...state.corrected] : [],
        comment: $("#fb-comment").value,
      }),
    });
    $("#fb-form").hidden = true;
    $("#fb-done").hidden = false;
  } catch (e) {
    toast(`Could not save feedback: ${e.message}`);
  }
}

/* ---------------- monitoring ---------------- */
const kpi = (label, value, sub = "") =>
  `<div class="kpi"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div><div class="kpi-sub">${sub}</div></div>`;
const hbar = (label, value, max, text, color) =>
  `<div class="hbar"><span>${label}</span><div class="hbar-track"><div class="hbar-fill" style="width:${max ? (value / max) * 100 : 0}%;${color ? `background:${color}` : ""}"></div></div><span class="num">${text}</span></div>`;

async function loadMonitoring() {
  try {
    const [s, recent] = await Promise.all([api("/api/stats"), api("/api/predictions?limit=25")]);
    const t = s.total || 0;
    const urgentRate = t ? s.by_triage.urgent / t : 0;
    const fbTotal = s.feedback.agree + s.feedback.disagree;
    $("#kpis").innerHTML =
      kpi("Studies (24h)", t, "processed by the model") +
      kpi("Urgent rate", pct(urgentRate), `${s.by_triage.urgent} studies prioritised`) +
      kpi("Avg latency", `${s.avg_latency_ms} ms`, "end-to-end, incl. preprocessing") +
      kpi("Radiologist agreement", s.agreement_rate == null ? "–" : pct(s.agreement_rate), `${fbTotal} feedback entries`);

    const maxH = Math.max(1, ...s.per_hour);
    $("#hours").innerHTML = s.per_hour.map((n, i) => `<div title="${n} studies, hour ${i + 1}" style="height:${(n / maxH) * 100}%"></div>`).join("");

    const css = getComputedStyle(document.documentElement);
    const colors = { urgent: css.getPropertyValue("--urgent"), review: css.getPropertyValue("--review"), routine: css.getPropertyValue("--routine") };
    $("#triage-mix").innerHTML =
      `<div class="stack">${Object.entries(s.by_triage).map(([k, v]) => `<div style="width:${t ? (v / t) * 100 : 0}%;background:${colors[k]}"></div>`).join("")}</div>` +
      Object.entries(s.by_triage).map(([k, v]) => `<div class="mix-row"><span><i style="background:${colors[k]}"></i>${k}</span><b>${v}</b></div>`).join("");

    const flags = Object.entries(s.flagged_counts).sort((a, b) => b[1] - a[1]).slice(0, 8);
    const maxF = Math.max(1, ...flags.map((f) => f[1]));
    $("#flag-bars").innerHTML = flags.some((f) => f[1]) ? flags.map(([l, n]) => hbar(LABEL(l), n, maxF, n)).join("") : `<div class="empty">No flagged findings yet</div>`;

    const tbody = $("#recent tbody");
    tbody.innerHTML = recent.length
      ? recent.map((p) => `<tr>
          <td>${new Date(p.ts * 1000).toLocaleTimeString()}</td>
          <td class="mono">${p.id}</td>
          <td><span class="pill ${p.triage}">${p.triage}</span></td>
          <td>${LABEL(p.top_finding)}</td>
          <td>${p.flagged.map(LABEL).join(", ") || "–"}</td>
          <td class="mono">${p.latency_ms.toFixed(0)} ms</td>
          <td>${p.verdict ? (p.verdict === "agree" ? "✓ agree" : "✕ disagree") : "–"}</td></tr>`).join("")
      : `<tr><td colspan="7" class="empty">No predictions yet. Analyze an image first.</td></tr>`;
  } catch (e) {
    toast(`Could not load monitoring data: ${e.message}`);
  }
}
$("#refresh-btn").addEventListener("click", loadMonitoring);

/* ---------------- model card ---------------- */
async function renderModelCard() {
  try {
    const c = state.card || (state.card = await api("/api/model"));
    const m = c.metrics || {};
    const ci = m.auroc_macro_ci ? `95% CI ${m.auroc_macro_ci[0].toFixed(3)}–${m.auroc_macro_ci[1].toFixed(3)}` : "";
    $("#mc-sub").textContent = `${c.name} v${c.version} · ${c.backbone} · ${c.img_size}px input · trained on ${c.training_data}`;
    $("#mc-kpis").innerHTML =
      kpi("Macro AUROC", m.auroc_macro != null ? m.auroc_macro.toFixed(3) : "–", ci) +
      kpi("Urgent-findings AUROC", m.auroc_urgent_macro != null ? m.auroc_urgent_macro.toFixed(3) : "–", "6 time-critical findings") +
      kpi("Calibration error (ECE)", m.ece_calibrated != null ? m.ece_calibrated.toFixed(3) : "–", m.ece_uncalibrated != null ? `${m.ece_uncalibrated.toFixed(3)} before temperature scaling` : "") +
      kpi("Test images", m.n_test != null ? m.n_test.toLocaleString() : "–", `target sensitivity ${c.target_sensitivity ? pct(c.target_sensitivity) : "–"}`);
    const auc = Object.entries(c.per_class_auroc || {}).filter(([, v]) => v != null).sort((a, b) => b[1] - a[1]);
    $("#mc-auc").innerHTML = auc.length
      ? auc.map(([l, v]) => hbar(LABEL(l), Math.max(0, v - 0.5), 0.5, v.toFixed(3))).join("")
      : `<div class="empty">No evaluation metrics in this bundle</div>`;
    $("#mc-use").textContent = c.intended_use;
    const prov = { Version: c.version, Backbone: c.backbone, Temperature: c.temperature?.toFixed(3), "MLflow run": c.mlflow_run_id || "–", Created: c.created_at, "ONNX SHA-256": c.onnx_sha256?.slice(0, 16) + "…" };
    $("#mc-prov").innerHTML = Object.entries(prov).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  } catch (e) {
    toast(`Could not load model card: ${e.message}`);
  }
}

/* ---------------- boot ---------------- */
$("#triage-card").dataset.placeholder = $("#triage-card").innerHTML;
checkHealth();
setInterval(checkHealth, 30000);
