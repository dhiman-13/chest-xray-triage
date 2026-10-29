"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const LABEL = (l) => l.replace(/_/g, " ");
const pct = (p) => `${(p * 100).toFixed(1)}%`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const ICONS = {
  urgent: '<svg viewBox="0 0 24 24"><path d="M12 8v5M12 16.5v.01"/><path d="M10.3 3.9 2.6 17.5A2 2 0 0 0 4.3 20.5h15.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>',
  review: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.4-4.4"/></svg>',
  routine: '<svg viewBox="0 0 24 24"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg>',
};
const TRIAGE_COPY = {
  urgent: { title: "Prioritise for an immediate read", text: "A time-critical finding is above its operating point." },
  review: { title: "Abnormal, routine review", text: "Non-urgent findings are above their operating points." },
  routine: { title: "Nothing above threshold", text: "Can stay in the routine worklist. A radiologist still reads every study." },
};

const state = { result: null, selected: null, image: null, card: null, verdict: null, corrected: new Set(), overlay: true, sample: null };

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
  toast._t = setTimeout(() => (t.hidden = true), 3800);
}

/* ---------------- navigation ---------------- */
// "home" and "analyze" are the same page: home is its top, analyze its upload workspace
function show(view) {
  $$(".tab").forEach((b) => {
    const on = b.dataset.view === view;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", on);
  });
  const page = view === "home" ? "analyze" : view;
  const wasActive = $(`#view-${page}`).classList.contains("active");
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${page}`));
  if (view === "monitor") loadMonitoring();
  if (view === "model") renderModelCard();
  if (view === "analyze") $("#workspace").scrollIntoView({ behavior: wasActive ? "smooth" : "auto" });
  else window.scrollTo({ top: 0, behavior: wasActive ? "smooth" : "auto" });
}
$$("[data-view]").forEach((el) => el.addEventListener("click", (e) => { e.preventDefault(); show(el.dataset.view); }));

/* ---------------- theme ---------------- */
function setTheme(theme) {
  const light = theme === "light";
  if (light) document.documentElement.dataset.theme = "light";
  else delete document.documentElement.dataset.theme;
  $("#theme-btn").setAttribute("aria-label", `Switch to ${light ? "dark" : "light"} theme`);
  document.querySelector('meta[name="theme-color"]').content = light ? "#f4f7fb" : "#050a13";
  try { localStorage.setItem("radiassist-theme", theme); } catch (_) { /* private mode: theme just isn't remembered */ }
  if ($("#view-monitor").classList.contains("active")) loadMonitoring(); // chart colours come from CSS vars
}
$("#theme-btn").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light"));
setTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");

/* ---------------- health ---------------- */
async function checkHealth() {
  const el = $("#status");
  try {
    const h = await api("/health");
    el.className = `status ${h.status === "ok" ? "ok" : ""}`;
    $("#status-text").textContent = h.status === "ok" ? "Model online" : "Model loading";
    $("#footer-version").textContent = h.model_version ? `Model ${h.model_version}` : "";
    if (!state.card && h.status === "ok") {
      state.card = await api("/api/model");
      const m = state.card.metrics || {};
      if (m.auroc_macro != null) $("#stat-auroc").textContent = m.auroc_macro.toFixed(3);
      if (m.n_test != null) $("#stat-ntest").textContent = m.n_test.toLocaleString();
    }
  } catch (_) {
    el.className = "status err";
    $("#status-text").textContent = "API unreachable";
  }
}

/* ---------------- samples ---------------- */
async function loadSamples() {
  try {
    const samples = await (await fetch("samples/samples.json")).json();
    $("#sample-grid").innerHTML = samples.map((s, i) => `
      <button class="sample" type="button" data-i="${i}" title="${esc(s.label)} · ${esc(s.view)} view · ${s.age}${esc(s.sex)}">
        <img src="samples/${esc(s.file)}" alt="Chest X-ray, ${esc(s.label)}" loading="lazy" width="512" height="512">
        <span>${esc(LABEL(s.label))}</span>
      </button>`).join("");
    $$(".sample").forEach((b) => b.addEventListener("click", () => analyzeSample(samples[b.dataset.i])));
  } catch (_) {
    $("#samples").hidden = true;
  }
}

async function analyzeSample(s) {
  $$(".sample").forEach((x) => x.querySelector("img").getAttribute("src").endsWith(s.file) ? x.classList.add("active") : x.classList.remove("active"));
  const blob = await (await fetch(`samples/${s.file}`)).blob();
  analyze(new File([blob], s.file, { type: "image/jpeg" }), s);
}

/* ---------------- hero ---------------- */
const toWorkspace = () => $("#workspace").scrollIntoView({ behavior: "smooth" });
// open the file picker straight from the hero; analyze() scrolls to the results once a file is chosen
$("#cta-upload").addEventListener("click", () => input.click());

/* ---------------- demo ----------------
   Replays the deployed model's real outputs (precomputed in samples/demo.json) on NIH test
   images: scan -> localize (heatmap) -> score -> triage. Pauses on hover and when off-screen. */
const demo = { items: [], i: 0, timers: [], visible: true, hover: false };
const DEMO_MS = 7000;

async function loadDemo() {
  try {
    const d = await (await fetch("samples/demo.json")).json();
    // the clearest teaching cases first: reported finding detected and ranked on top
    demo.items = d.samples.filter((s) => s.label !== "No Finding")
      .sort((a, b) => (b.findings[0].label === b.label) - (a.findings[0].label === a.label));
    demo.items.forEach((s) => { const im = new Image(); im.src = `samples/${s.file}`; });
    $("#demo-dots").innerHTML = demo.items.map((s, i) =>
      `<button type="button" role="tab" aria-label="${esc(LABEL(s.label))}" data-i="${i}"></button>`).join("");
    $$("#demo-dots button").forEach((b) => b.addEventListener("click", () => playDemo(+b.dataset.i)));
    $("#demo-try").addEventListener("click", () => { toWorkspace(); analyzeSample(demo.items[demo.i]); });
    const el = $("#demo");
    el.addEventListener("mouseenter", () => { demo.hover = true; el.classList.add("paused"); });
    el.addEventListener("mouseleave", () => { demo.hover = false; el.classList.remove("paused"); });
    new IntersectionObserver(([e]) => { demo.visible = e.isIntersecting; }).observe(el);
    playDemo(0);
  } catch (_) {
    $("#demo").hidden = true;
  }
}

function demoAt(ms, fn) { demo.timers.push(setTimeout(fn, ms)); }

function playDemo(i) {
  demo.timers.forEach(clearTimeout);
  demo.timers = [];
  demo.i = i;
  const s = demo.items[i];
  const el = $("#demo");
  const steps = $$("#demo-steps li");
  const setStep = (k) => steps.forEach((li, j) => { li.classList.toggle("on", j === k); li.classList.toggle("done", j < k); });

  el.className = "demo";
  if (demo.hover) el.classList.add("paused");
  setStep(-1);
  $$("#demo-dots button").forEach((b, j) => {
    b.classList.remove("on");
    if (j === i) { void b.offsetWidth; b.classList.add("on"); b.style.setProperty("--dur", `${DEMO_MS}ms`); }
  });
  $("#demo-src").textContent = `NIH test image · ${s.view} view · ${s.age}${s.sex}`;
  const img = $("#demo-img");
  img.src = `samples/${s.file}`;
  img.alt = `Chest X-ray reported as ${LABEL(s.label)}`;
  renderHeat(s.heatmap, $("#demo-heat"));
  const focus = s.findings.find((f) => f.label === s.focus);
  $("#demo-tag").textContent = `Looking at: ${LABEL(s.focus)}`;
  $("#demo-findings").innerHTML = s.findings.map((f) => `
    <li class="${f.positive ? "pos" : ""} ${f.urgent ? "urgent" : ""} ${f.label === s.focus ? "focus" : ""}">
      <span class="n">${LABEL(f.label)}</span><span class="v">${pct(f.probability)}</span>
      <span class="b"><i data-w="${(f.probability * 100).toFixed(1)}"></i></span>
    </li>`).join("");
  const detected = focus ? focus.positive : s.findings.some((f) => f.label === s.label && f.positive);
  $("#demo-verdict").innerHTML = `
    <span class="verdict-chip ${s.triage}">${ICONS[s.triage]}${s.triage}</span>
    <p class="verdict-truth">Radiologist report: <b>${esc(LABEL(s.label))}</b> ·
      ${detected ? '<span class="ok">detected</span>' : "not flagged"} · ${s.flagged} of 14 flagged</p>`;

  demoAt(50, () => { el.classList.add("s-in", "s-scan"); setStep(0); });
  demoAt(1600, () => { el.classList.add("s-loc"); setStep(1); });
  // absolute times, and each step also completes the earlier ones, so throttled timers
  // (background tabs, slow phones) can never leave the demo half drawn
  const showBars = (n) => $$("#demo-findings li").forEach((li, k) => {
    if (k >= n) return;
    li.classList.add("show");
    const bar = li.querySelector(".b i");
    bar.style.width = `${bar.dataset.w}%`;
  });
  demoAt(2400, () => { el.classList.add("s-in", "s-loc"); setStep(2); });
  s.findings.forEach((_, k) => demoAt(2400 + k * 180, () => showBars(k + 1)));
  demoAt(3700, () => { el.classList.add("s-in", "s-loc", "s-tri"); showBars(Infinity); setStep(3); });
  demoAt(DEMO_MS, function next() {
    if (demo.hover || !demo.visible || document.hidden) return demoAt(500, next);
    playDemo((demo.i + 1) % demo.items.length);
  });
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
  state.result = state.sample = null;
  input.value = "";
  $("#stage").hidden = true;
  $("#viewer-tools").hidden = true;
  dz.hidden = false;
  $("#intro-card").hidden = false;
  ["#triage-card", "#findings-card", "#feedback-card"].forEach((s) => ($(s).hidden = true));
  $$(".sample").forEach((x) => x.classList.remove("active"));
}

async function analyze(file, sample = null) {
  if (!/image\/(png|jpeg)/.test(file.type)) return toast("Please upload a PNG or JPEG image");
  state.sample = sample;
  dz.hidden = true;
  const stage = $("#stage");
  stage.hidden = false;
  stage.classList.add("loading");
  $("#heat-caption").textContent = "";
  await drawImageFromFile(file);
  const mobile = window.matchMedia("(max-width: 980px)").matches;
  (mobile ? $("#viewer") : $("#workspace")).scrollIntoView({ behavior: "smooth", block: "start" });

  const form = new FormData();
  form.append("file", file);
  try {
    const [result] = await Promise.all([
      api("/api/predict", { method: "POST", body: form }),
      new Promise((r) => setTimeout(r, 700)), // let the scan animation read as deliberate
    ]);
    // The API ranks by probability / threshold, which over-promotes classes with tiny thresholds
    // (e.g. Hernia). Readers expect the most likely findings first: flagged ones, by probability.
    result.findings.sort((a, b) => (b.positive - a.positive) || (b.probability - a.probability));
    state.result = result;
    state.image = await loadImage(result.image);
    const reported = sample && result.findings.find((f) => f.label === sample.label);
    state.selected = (reported || result.findings[0]).label;
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
  try {
    const img = await loadImage(url);
    const c = $("#canvas"), ctx = c.getContext("2d");
    ctx.fillStyle = "#050607";
    ctx.fillRect(0, 0, c.width, c.height);
    const s = Math.min(c.width / img.width, c.height / img.height);
    ctx.drawImage(img, (c.width - img.width * s) / 2, (c.height - img.height * s) / 2, img.width * s, img.height * s);
  } finally {
    URL.revokeObjectURL(url);
  }
}

/* ---------------- heatmap ---------------- */
// Polynomial approximation of Google's Turbo colormap
function turbo(t) {
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return [r, g, b].map((v) => Math.round(255 * Math.min(1, Math.max(0, v))));
}

const heatCache = new Map();
function heatCanvas(label) {
  if (!heatCache.has(label)) {
    const c = document.createElement("canvas");
    c.width = c.height = 192;
    renderHeat(state.result.heatmaps[label], c);
    heatCache.set(label, c);
  }
  return heatCache.get(label);
}

function renderHeat(grid, c) {
  const size = c.width;
  const h = grid.length, w = grid[0].length;
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
}

function drawStage() {
  if (!state.image || !state.result) return;
  const c = $("#canvas"), ctx = c.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.globalAlpha = 1;
  ctx.drawImage(state.image, 0, 0, c.width, c.height);
  $("#stage").classList.toggle("no-overlay", !state.overlay);
  const f = state.result.findings.find((x) => x.label === state.selected);
  if (state.overlay && f) {
    ctx.globalAlpha = $("#heat-opacity").value / 100;
    ctx.drawImage(heatCanvas(f.label), 0, 0, c.width, c.height);
    ctx.globalAlpha = 1;
    $("#heat-caption").textContent = `${LABEL(f.label)} · ${pct(f.probability)}`;
  } else {
    $("#heat-caption").textContent = "";
  }
}
$("#heat-opacity").addEventListener("input", drawStage);
$$(".seg").forEach((b) => b.addEventListener("click", () => {
  state.overlay = b.dataset.overlay === "on";
  $$(".seg").forEach((x) => x.classList.toggle("active", x === b));
  drawStage();
}));

/* ---------------- results ---------------- */
function renderResult() {
  const r = state.result;
  heatCache.clear();
  const copy = TRIAGE_COPY[r.triage];
  const flagged = r.findings.filter((f) => f.positive);
  const urgent = flagged.filter((f) => f.urgent).map((f) => LABEL(f.label));
  const sub = r.triage === "urgent" ? `Above operating point: ${urgent.join(", ")}.`
    : r.triage === "review" ? `Above operating point: ${flagged.map((f) => LABEL(f.label)).join(", ")}.`
    : copy.text;

  let truth = "";
  if (state.sample) {
    const label = state.sample.label;
    const match = r.findings.find((f) => f.label === label);
    const hit = label === "No Finding" ? r.triage === "routine" : Boolean(match?.positive);
    const verdict = label === "No Finding" ? (hit ? "Correctly routine" : "Over-flagged") : (hit ? "Detected" : "Missed");
    truth = `<div class="truth">Report label <b>${esc(LABEL(label))}</b> · ${esc(state.sample.view)} view, ${state.sample.age}${esc(state.sample.sex)}
      <span class="match ${hit ? "hit" : "miss"}">${verdict}</span></div>`;
  }

  const card = $("#triage-card");
  card.innerHTML = `
    <div class="triage ${r.triage}">
      <div class="triage-badge">${ICONS[r.triage]}</div>
      <div>
        <div class="triage-level">${r.triage}</div>
        <div class="triage-title">${copy.title}</div>
        <div class="triage-sub">${esc(sub)}</div>
      </div>
    </div>
    ${truth}
    <div class="triage-meta">
      <span>Flagged <b>${flagged.length}/14</b></span>
      <span>Latency <b>${Math.round(r.latency_ms)} ms</b></span>
      <span>Study <b>${esc(r.prediction_id)}</b></span>
    </div>`;

  const list = $("#findings");
  list.innerHTML = "";
  r.findings.forEach((f) => {
    const li = document.createElement("li");
    li.className = `finding${f.positive ? " positive" : ""}${f.urgent ? " is-urgent" : ""}${f.label === state.selected ? " selected" : ""}`;
    li.tabIndex = 0;
    li.setAttribute("role", "button");
    li.innerHTML = `
      <div class="f-name"><span>${LABEL(f.label)}</span>
        ${f.urgent ? '<i class="urgent-dot" title="Time-critical finding"></i>' : ""}
        ${f.positive ? `<span class="tag ${f.urgent ? "urgent" : "flag"}">${f.urgent ? "urgent" : "flagged"}</span>` : ""}
      </div>
      <div class="f-pct">${pct(f.probability)}</div>
      <div class="f-bar"><div class="f-fill"></div><div class="f-th" style="left:${Math.min(f.threshold, 1) * 100}%" title="Operating point ${pct(f.threshold)}"></div></div>`;
    const select = () => {
      state.selected = f.label;
      list.querySelectorAll(".finding").forEach((n) => n.classList.remove("selected"));
      li.classList.add("selected");
      if (!state.overlay) $$(".seg")[0].click();
      else drawStage();
    };
    li.addEventListener("click", select);
    li.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(); } });
    list.appendChild(li);
    requestAnimationFrame(() => requestAnimationFrame(() => (li.querySelector(".f-fill").style.width = `${f.probability * 100}%`)));
  });

  $("#intro-card").hidden = true;
  [card, $("#findings-card")].forEach((el) => (el.hidden = false));
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
  $("#fb-comment").value = "";
  $$(".fb-buttons .btn").forEach((b) => b.classList.remove("selected"));
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

$$(".fb-buttons .btn").forEach((btn) =>
  btn.addEventListener("click", async () => {
    $$(".fb-buttons .btn").forEach((b) => b.classList.toggle("selected", b === btn));
    state.verdict = btn.dataset.verdict;
    if (state.verdict === "disagree") {
      $("#fb-form").hidden = false;
      $("#fb-done").hidden = true;
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
  `<div class="hbar"><span title="${label}">${label}</span><div class="hbar-track"><div class="hbar-fill" style="width:${max ? (value / max) * 100 : 0}%;${color ? `background:${color}` : ""}"></div></div><span class="num">${text}</span></div>`;
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

async function loadMonitoring() {
  try {
    const [s, recent] = await Promise.all([api("/api/stats"), api("/api/predictions?limit=25")]);
    const t = s.total || 0;
    const fbTotal = s.feedback.agree + s.feedback.disagree;
    $("#kpis").innerHTML =
      kpi("Studies, last 24 h", t.toLocaleString(), "processed by the model") +
      kpi("Urgent rate", t ? pct(s.by_triage.urgent / t) : "–", `${s.by_triage.urgent} studies prioritised`) +
      kpi("Average latency", t ? `${Math.round(s.avg_latency_ms)} ms` : "–", "server side, including preprocessing") +
      kpi("Reader agreement", s.agreement_rate == null ? "–" : pct(s.agreement_rate), `${fbTotal} feedback ${fbTotal === 1 ? "entry" : "entries"}`);

    const maxH = Math.max(1, ...s.per_hour);
    $("#hours").innerHTML = s.per_hour.map((n, i) => `<div title="${n} studies" style="height:${(n / maxH) * 100}%"></div>`).join("");

    const colors = { urgent: cssVar("--urgent"), review: cssVar("--review"), routine: cssVar("--routine") };
    $("#triage-mix").innerHTML =
      `<div class="stack">${Object.entries(s.by_triage).map(([k, v]) => `<div style="width:${t ? (v / t) * 100 : 0}%;background:${colors[k]}"></div>`).join("")}</div>` +
      Object.entries(s.by_triage).map(([k, v]) => `<div class="mix-row"><span><i style="background:${colors[k]}"></i>${k}</span><span><b>${v}</b><em>${t ? pct(v / t) : ""}</em></span></div>`).join("");

    const flags = Object.entries(s.flagged_counts).sort((a, b) => b[1] - a[1]).slice(0, 7);
    const maxF = Math.max(1, ...flags.map((f) => f[1]));
    $("#flag-bars").innerHTML = flags.some((f) => f[1]) ? flags.map(([l, n]) => hbar(LABEL(l), n, maxF, n)).join("") : `<div class="empty">No flagged findings yet</div>`;

    $("#recent tbody").innerHTML = recent.length
      ? recent.map((p) => `<tr>
          <td>${new Date(p.ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</td>
          <td class="mono">${esc(p.id)}</td>
          <td><span class="pill ${p.triage}">${p.triage}</span></td>
          <td>${LABEL(p.top_finding)}</td>
          <td class="flagged">${p.flagged.map(LABEL).join(", ") || "–"}</td>
          <td class="mono r">${p.latency_ms.toFixed(0)} ms</td>
          <td>${p.verdict ? (p.verdict === "agree" ? "Agree" : "Disagree") : "–"}</td></tr>`).join("")
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
    const ci = m.auroc_macro_ci ? `95% CI ${m.auroc_macro_ci[0].toFixed(3)} – ${m.auroc_macro_ci[1].toFixed(3)}` : "";
    $("#mc-title").textContent = `${c.backbone === "densenet121" ? "DenseNet-121" : c.backbone}, version ${c.version}`;
    $("#mc-sub").textContent = `${c.img_size} px input · 14 findings · trained on ${c.training_data}.`;
    $("#mc-kpis").innerHTML =
      kpi("Macro AUROC", m.auroc_macro != null ? m.auroc_macro.toFixed(3) : "–", ci) +
      kpi("Urgent-findings AUROC", m.auroc_urgent_macro != null ? m.auroc_urgent_macro.toFixed(3) : "–", "6 time-critical findings") +
      kpi("Calibration error", m.ece_calibrated != null ? m.ece_calibrated.toFixed(3) : "–", m.ece_uncalibrated != null ? `ECE, down from ${m.ece_uncalibrated.toFixed(3)}` : "") +
      kpi("Test images", m.n_test != null ? m.n_test.toLocaleString() : "–", `operating points at ${c.target_sensitivity ? pct(c.target_sensitivity) : "–"} sensitivity`);
    const auc = Object.entries(c.per_class_auroc || {}).filter(([, v]) => v != null).sort((a, b) => b[1] - a[1]);
    $("#mc-auc").innerHTML = auc.length
      ? auc.map(([l, v]) => hbar(LABEL(l), Math.max(0, v - 0.5), 0.5, v.toFixed(3))).join("")
      : `<div class="empty">No evaluation metrics in this bundle</div>`;
    $("#mc-use").textContent = c.intended_use;
    const prov = { Version: c.version, Backbone: c.backbone, Temperature: c.temperature?.toFixed(3), "MLflow run": c.mlflow_run_id || "–", Created: c.created_at, "ONNX SHA-256": c.onnx_sha256 ? `${c.onnx_sha256.slice(0, 16)}…` : "–" };
    $("#mc-prov").innerHTML = Object.entries(prov).map(([k, v]) => `<dt>${k}</dt><dd>${esc(v ?? "–")}</dd>`).join("");
  } catch (e) {
    toast(`Could not load model card: ${e.message}`);
  }
}

/* ---------------- boot ---------------- */
loadSamples();
loadDemo();
checkHealth();
setInterval(checkHealth, 30000);
