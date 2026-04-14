// TransBench Analytics: Comparative Analysis Dashboard
// - Efficiency Frontier scatter (loss vs tokens/s)
// - Dense pivot table with heatmap + optional baseline deltas
// - Convergence chart (loss/lr series) over steps

const MANIFEST_URL = "/reports/manifest.json";

const SHELL = document.querySelector(".app-shell");

const UI = {
  refresh: document.getElementById("refresh"),
  dataset: document.getElementById("dataset"),
  deviceToggle: document.getElementById("deviceToggle"),
  aggMode: document.getElementById("aggMode"),
  baselineArch: document.getElementById("baselineArch"),
  seriesMetric: document.getElementById("seriesMetric"),
  archList: document.getElementById("archList"),
  archAll: document.getElementById("archAll"),
  archNone: document.getElementById("archNone"),
  sidebar: document.getElementById("sidebar"),
  sidebarToggle: document.getElementById("sidebarToggle"),
  tableSubtitle: document.getElementById("tableSubtitle"),
  compTable: document.getElementById("compTable"),
  scatterCanvas: document.getElementById("scatterChart"),
  lineCanvas: document.getElementById("lineChart"),
  scatterEmpty: document.getElementById("scatterEmpty"),
  lineEmpty: document.getElementById("lineEmpty"),
  tableEmpty: document.getElementById("tableEmpty"),
  // Tab 4: Noise Crossover
  crossoverDatasetToggle: document.getElementById("crossoverDatasetToggle"),
  crossoverCards: document.getElementById("crossoverCards"),
  crossoverLineCanvas: document.getElementById("crossoverLineChart"),
  crossoverDeltaCanvas: document.getElementById("crossoverDeltaChart"),
  crossoverEmpty: document.getElementById("crossoverEmpty"),
  // Tab 5: Gate Dynamics
  gateArchToggle: document.getElementById("gateArchToggle"),
  gateEvoCanvas: document.getElementById("gateEvoChart"),
  auxLossCanvas: document.getElementById("auxLossChart"),
  gatesEmpty: document.getElementById("gatesEmpty"),
  // Tab 6: Training Stability
  timingCanvas: document.getElementById("timingChart"),
  memoryCanvas: document.getElementById("memoryChart"),
  evalGapCanvas: document.getElementById("evalGapChart"),
  paramBreakdownCanvas: document.getElementById("paramBreakdownChart"),
  stabilityEmpty: document.getElementById("stabilityEmpty"),
  // Tab 7: Radar
  radarCanvas: document.getElementById("radarChart"),
  radarBestTable: document.getElementById("radarBestTable"),
  radarEmpty: document.getElementById("radarEmpty"),
  // Tab 8: Run Inspector
  inspectorRunSelect: document.getElementById("inspectorRunSelect"),
  inspectorCards: document.getElementById("inspectorCards"),
  inspectorLossCanvas: document.getElementById("inspectorLossChart"),
  inspectorGateCanvas: document.getElementById("inspectorGateChart"),
  inspectorConfigTable: document.getElementById("inspectorConfigTable"),
  inspectorEmpty: document.getElementById("inspectorEmpty"),
};

const state = {
  manifest: null,
  charts: { scatter: null, line: null },
  newCharts: {},
  reportCache: new Map(),
};

function getTabs() {
  const tabs = Array.from(document.querySelectorAll(".tab[role=tab]"));
  const panels = Array.from(document.querySelectorAll(".tab-panel[role=tabpanel]"));
  return { tabs, panels };
}

function activateTabById(tabId) {
  const { tabs, panels } = getTabs();
  const tab = tabs.find((t) => t.id === tabId);
  if (!tab) return;

  for (const t of tabs) {
    const active = t.id === tabId;
    t.classList.toggle("is-active", active);
    t.setAttribute("aria-selected", active ? "true" : "false");
    t.tabIndex = active ? 0 : -1;
  }

  for (const p of panels) {
    const active = p.id === tab.getAttribute("aria-controls");
    p.classList.toggle("is-active", active);
    if (active) {
      p.removeAttribute("hidden");
    } else {
      p.setAttribute("hidden", "");
    }
  }

  // Ensure charts created while hidden get proper sizing when shown.
  if (tabId === "tab-frontier") {
    state.charts.scatter?.resize();
    state.charts.scatter?.update();
  } else if (tabId === "tab-convergence") {
    state.charts.line?.resize();
    state.charts.line?.update();
  }
  // Resize new-tab charts
  for (const c of Object.values(state.newCharts)) {
    if (c) { c.resize(); c.update(); }
  }

  // Lazy render for new tabs
  const picked = getPicked();
  if (tabId === "tab-crossover") renderNoiseCrossover().catch(console.error);
  if (tabId === "tab-gates") renderGateDynamics().catch(console.error);
  if (tabId === "tab-stability") renderStability(picked).catch(console.error);
  if (tabId === "tab-radar") renderRadar(picked).catch(console.error);
  if (tabId === "tab-inspector") { fillInspectorSelect(); renderInspector().catch(console.error); }
}

function wireTabs() {
  const { tabs } = getTabs();
  if (!tabs.length) return;

  for (const tab of tabs) {
    tab.addEventListener("click", () => {
      activateTabById(tab.id);
    });

    tab.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight" && e.key !== "Home" && e.key !== "End") return;
      e.preventDefault();
      const idx = tabs.findIndex((t) => t.id === tab.id);
      let next = idx;
      if (e.key === "ArrowLeft") next = (idx - 1 + tabs.length) % tabs.length;
      if (e.key === "ArrowRight") next = (idx + 1) % tabs.length;
      if (e.key === "Home") next = 0;
      if (e.key === "End") next = tabs.length - 1;
      tabs[next]?.focus();
      activateTabById(tabs[next].id);
    });
  }
}

const SERIES_OPTS = [
  { value: "loss_series", label: "Loss Curve" },
  { value: "lr_series", label: "LR Schedule" },
  { value: "loss_lr", label: "Loss + LR" },
];

const METRICS_CFG = [
  {
    id: "tokens_per_s",
    label: "Throughput (tok/s)",
    better: "higher",
    format: (n) => Number(n).toFixed(1),
  },
  {
    id: "loss_mean",
    label: "Loss (eval)",
    better: "lower",
    format: (n) => Number(n).toFixed(4),
  },
  {
    id: "peak_mem_mb",
    label: "Peak Memory (MB)",
    better: "lower",
    format: (n) => Number(n).toFixed(0),
  },
  {
    id: "train_step_ms_mean",
    label: "Step Time (ms)",
    better: "lower",
    format: (n) => Number(n).toFixed(2),
  },
  {
    id: "forward_ms_mean",
    label: "Forward Time (ms)",
    better: "lower",
    format: (n) => Number(n).toFixed(2),
  },
  {
    id: "params",
    label: "Params (M)",
    better: "none",
    format: (n) => (Number(n) / 1e6).toFixed(2),
  },
  {
    id: "tokens_per_s_per_mparam",
    label: "Tok/s per MParam",
    better: "higher",
    format: (n) => Number(n).toFixed(2),
  },
  {
    id: "learning_rate",
    label: "Learning Rate",
    better: "none",
    format: (n) => Number(n).toExponential(2),
  },
];

// Allow all datasets found in the manifest to appear in the dropdown
const ALLOWED_DATASETS = ["tinystories", "poisoned_needle", "sparse_signal", "zeros", "ramp", "tinystories-instruct", "wikitext"];

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[m]));
}

function uniq(arr) {
  return Array.from(new Set(arr));
}

function toNum(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function getArchColor(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) hash = str.charCodeAt(i) + ((hash << 5) - hash);
  const hue = Math.abs(hash % 360);
  return `hsl(${hue}, 70%, 45%)`;
}

function getReports() {
  return Array.isArray(state.manifest?.reports) ? state.manifest.reports : [];
}

function selectedDataset() {
  return UI.dataset?.value || "all";
}

function selectedDevice() {
  const active = UI.deviceToggle?.querySelector(".seg-btn.is-active");
  return active?.dataset.value || "all";
}

function filteredReports() {
  const reports = getReports();
  const ds = selectedDataset();
  const dev = selectedDevice();
  return reports.filter((r) => {
    if (ds && ds !== "all" && (r.dataset || "unknown") !== ds) return false;
    if (dev && dev !== "all" && (r.device || "cpu") !== dev) return false;
    return true;
  });
}

function selectedArchs() {
  if (!UI.archList) return [];
  const checked = UI.archList.querySelectorAll("input[type=checkbox]:checked");
  return Array.from(checked).map((n) => n.value).filter(Boolean);
}

function datasetCounts(reports) {
  const counts = new Map();
  for (const r of reports) {
    const d = r.dataset || "unknown";
    counts.set(d, (counts.get(d) || 0) + 1);
  }
  return counts;
}

function datasetValues(reports) {
  return uniq(reports.map((r) => r.dataset || "unknown")).sort();
}

function archValues(reports) {
  return uniq(reports.map((r) => r.arch).filter(Boolean)).sort();
}

async function loadManifest() {
  const res = await fetch(MANIFEST_URL, { cache: "no-store" });
  if (!res.ok) throw new Error(`Cannot load ${MANIFEST_URL}. Start 'serve-dashboard' first.`);
  state.manifest = await res.json();
}

function fillDatasetSelect() {
  const reports = getReports();
  const counts = datasetCounts(reports);
  const present = datasetValues(reports).filter((d) => ALLOWED_DATASETS.includes(d));
  const prev = UI.dataset.value;

  UI.dataset.innerHTML = "";

  const optAll = document.createElement("option");
  optAll.value = "all";
  optAll.textContent = "All datasets";
  UI.dataset.appendChild(optAll);

  if (present.length) {
    const g = document.createElement("optgroup");
    g.label = "Datasets in reports";
    for (const d of present) {
      const n = counts.get(d) || 0;
      const o = document.createElement("option");
      o.value = d;
      o.textContent = `${d} (${n})`;
      g.appendChild(o);
    }
    UI.dataset.appendChild(g);
  }

  const values = new Set(Array.from(UI.dataset.querySelectorAll("option")).map((o) => o.value));
  if (values.has(prev)) {
    UI.dataset.value = prev;
  } else {
    // Default: prefer 'tinystories' if available, otherwise pick the dataset
    // with the most reports to reduce confusion.
    let best = "all";
    let bestCount = 0;
    if (present.includes("tinystories")) {
      best = "tinystories";
    } else {
      for (const d of present) {
        const n = counts.get(d) || 0;
        if (n > bestCount) {
          best = d;
          bestCount = n;
        }
      }
    }
    UI.dataset.value = values.has(best) ? best : "all";
  }
}

function fillSeriesSelect() {
  const prev = UI.seriesMetric.value;
  UI.seriesMetric.innerHTML = "";
  for (const opt of SERIES_OPTS) {
    const o = document.createElement("option");
    o.value = opt.value;
    o.textContent = opt.label;
    UI.seriesMetric.appendChild(o);
  }
  UI.seriesMetric.value = prev || "loss_series";
}

function renderArchList() {
  const reports = filteredReports();
  const archs = archValues(reports);
  const prev = new Set(selectedArchs());
  const keepPrev = prev.size > 0;

  UI.archList.innerHTML = "";
  for (const arch of archs) {
    const row = document.createElement("label");
    row.className = "arch-item";
    const checked = keepPrev ? prev.has(arch) : true;
    const color = getArchColor(arch);
    row.innerHTML = `
      <input type="checkbox" value="${escapeHtml(arch)}" ${checked ? "checked" : ""} />
      <span style="color:${color}; font-weight:800;">●</span>
      <span>${escapeHtml(arch)}</span>
    `;
    UI.archList.appendChild(row);
  }

  fillBaselineSelect(archs);
}

function fillBaselineSelect(archs) {
  const prev = UI.baselineArch.value;
  UI.baselineArch.innerHTML = "";

  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = "— Absolute Values —";
  UI.baselineArch.appendChild(empty);

  for (const arch of archs) {
    const o = document.createElement("option");
    o.value = arch;
    o.textContent = arch;
    UI.baselineArch.appendChild(o);
  }

  if (prev && archs.includes(prev)) UI.baselineArch.value = prev;
}

function aggregateReports(reports, archs) {
  const mode = UI.aggMode.value;

  // reports come sorted by timestamp desc in manifest. Preserve this order.
  const perArch = new Map();
  for (const arch of archs) perArch.set(arch, []);
  for (const r of reports) {
    if (!r.arch || !perArch.has(r.arch)) continue;
    perArch.get(r.arch).push(r);
  }

  const picked = [];
  for (const arch of archs) {
    const runs = perArch.get(arch) || [];
    if (!runs.length) continue;

    if (mode === "latest") {
      picked.push({ arch, run: runs[0] });
      continue;
    }

    // mode === best : min loss_mean
    let best = null;
    let bestLoss = Infinity;
    for (const r of runs) {
      const loss = toNum(r.loss_mean);
      if (loss === null) continue;
      if (loss < bestLoss) {
        bestLoss = loss;
        best = r;
      }
    }
    picked.push({ arch, run: best || runs[0] });
  }
  return picked;
}

function getPicked() {
  const reports = filteredReports();
  const allArchs = archValues(reports);
  const chosen = selectedArchs();
  const archs = chosen.length ? chosen : allArchs;
  return aggregateReports(reports, archs);
}

function setEmpty(el, isEmpty) {
  if (!el) return;
  el.hidden = !isEmpty;
}

function renderScatter(picked) {
  const points = picked
    .map((p) => {
      const x = toNum(p.run.tokens_per_s);
      const y = toNum(p.run.loss_mean);
      if (x === null || y === null) return null;
      return { arch: p.arch, x, y };
    })
    .filter(Boolean);

  setEmpty(UI.scatterEmpty, points.length === 0);
  if (points.length === 0) {
    if (state.charts.scatter) {
      state.charts.scatter.data.datasets = [];
      state.charts.scatter.update();
    }
    return;
  }

  const datasets = points.map((pt) => {
    const color = getArchColor(pt.arch);
    return {
      label: pt.arch,
      data: [{ x: pt.x, y: pt.y }],
      backgroundColor: color,
      borderColor: color,
      pointRadius: 6,
      pointHoverRadius: 7,
    };
  });

  if (!state.charts.scatter) {
    state.charts.scatter = new Chart(UI.scatterCanvas.getContext("2d"), {
      type: "scatter",
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const arch = ctx.dataset.label;
                const x = ctx.raw?.x;
                const y = ctx.raw?.y;
                const xTxt = Number.isFinite(Number(x)) ? Number(x).toFixed(0) : "—";
                const yTxt = Number.isFinite(Number(y)) ? Number(y).toFixed(4) : "—";
                return `${arch}: loss ${yTxt} @ ${xTxt} tok/s`;
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "Throughput (tokens/s) →" },
            grid: { color: "#e5e7eb" },
          },
          y: {
            title: { display: true, text: "Loss (lower is better) ↓" },
            grid: { color: "#e5e7eb" },
          },
        },
      },
    });
  } else {
    state.charts.scatter.data.datasets = datasets;
    state.charts.scatter.update();
  }
}

function metricExtrema(values, better) {
  const items = values
    .map((v, idx) => ({ v, idx }))
    .filter((x) => x.v !== null && Number.isFinite(x.v));
  if (items.length < 2) return { bestIdx: null, worstIdx: null };

  if (better === "lower") {
    items.sort((a, b) => a.v - b.v);
  } else {
    items.sort((a, b) => b.v - a.v);
  }
  return { bestIdx: items[0].idx, worstIdx: items[items.length - 1].idx };
}

function deltaFor(metric, val, base) {
  if (metric.better === "none") return null;
  if (base === 0) return null;
  if (metric.better === "higher") return ((val - base) / base) * 100;
  if (metric.better === "lower") return ((base - val) / base) * 100;
  return null;
}

function deltaClass(d) {
  if (d === null) return "neu";
  if (Math.abs(d) < 0.05) return "neu";
  return d > 0 ? "pos" : "neg";
}

function renderTable(picked) {
  const thead = UI.compTable.querySelector("thead");
  const tbody = UI.compTable.querySelector("tbody");
  tbody.innerHTML = "";

  if (!picked.length) {
    thead.innerHTML = "";
    setEmpty(UI.tableEmpty, true);
    return;
  }
  setEmpty(UI.tableEmpty, false);

  const baseline = UI.baselineArch.value || "";
  const baselineIdx = baseline ? picked.findIndex((p) => p.arch === baseline) : -1;
  const baselineRun = baselineIdx >= 0 ? picked[baselineIdx].run : null;

  // Header
  let head = `<tr><th>Metric</th>`;
  for (const p of picked) {
    const isBase = baseline && p.arch === baseline;
    const color = getArchColor(p.arch);
    head += `<th class="${isBase ? "baseline" : ""}" style="color:${color}">${escapeHtml(p.arch)}${
      isBase ? ' <span class="muted">(Base)</span>' : ""
    }</th>`;
  }
  head += `</tr>`;
  thead.innerHTML = head;

  // Rows
  for (const metric of METRICS_CFG) {
    const row = document.createElement("tr");
    const values = picked.map((p) => toNum(p.run?.[metric.id]));
    const { bestIdx, worstIdx } = metric.better === "none" ? { bestIdx: null, worstIdx: null } : metricExtrema(values, metric.better);

    let html = `<td>${escapeHtml(metric.label)}</td>`;
    for (let i = 0; i < picked.length; i++) {
      const v = values[i];
      const isBase = baselineIdx === i;
      const classes = [];
      if (isBase) classes.push("baseline");
      if (bestIdx === i) classes.push("bg-good");
      if (worstIdx === i) classes.push("bg-bad");

      if (v === null) {
        html += `<td class="${classes.join(" ")}"><span class="muted">—</span></td>`;
        continue;
      }

      const formatted = metric.format(v);
      let delta = "";
      if (baselineRun && !isBase) {
        const base = toNum(baselineRun?.[metric.id]);
        if (base !== null && base !== 0) {
          const d = deltaFor(metric, v, base);
          if (d !== null && Number.isFinite(d)) {
            const sign = d > 0 ? "+" : "";
            const cls = deltaClass(d);
            delta = ` <span class="delta ${cls}">${sign}${d.toFixed(1)}%</span>`;
          }
        }
      }

      html += `<td class="${classes.join(" ")}"><span class="val">${escapeHtml(formatted)}</span>${delta}</td>`;
    }
    row.innerHTML = html;
    tbody.appendChild(row);
  }
}

async function fetchReportJson(file) {
  if (state.reportCache.has(file)) return state.reportCache.get(file);
  const res = await fetch(`/reports/${encodeURIComponent(file)}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch report: ${file}`);
  const json = await res.json();
  state.reportCache.set(file, json);
  return json;
}

async function renderLine(picked) {
  const metricKey = UI.seriesMetric.value || "loss_series";

  if (!picked.length) {
    setEmpty(UI.lineEmpty, true);
    if (state.charts.line) {
      state.charts.line.data.labels = [];
      state.charts.line.data.datasets = [];
      state.charts.line.update();
    }
    return;
  }

  const payloads = await Promise.all(
    picked.map(async (p) => {
      try {
        const json = await fetchReportJson(p.run.file);
        return { arch: p.arch, json };
      } catch (e) {
        console.warn("Failed to load report", p.arch, e);
        return { arch: p.arch, json: null };
      }
    })
  );

  const wantsDual = metricKey === "loss_lr";
  const perArchKey = wantsDual ? "loss_series" : metricKey;

  let maxLen = 0;
  const series = payloads
    .map((p) => {
      const arr = p.json?.metrics?.[perArchKey];
      if (!Array.isArray(arr) || arr.length === 0) return null;
      const vals = arr.map((v) => (Number.isFinite(Number(v)) ? Number(v) : null));
      maxLen = Math.max(maxLen, vals.length);
      return { arch: p.arch, vals };
    })
    .filter(Boolean);

  let lrSeries = null;
  if (wantsDual) {
    const firstWithLr = payloads.find((p) => Array.isArray(p.json?.metrics?.lr_series) && p.json.metrics.lr_series.length);
    if (firstWithLr) {
      const arr = firstWithLr.json.metrics.lr_series;
      const vals = arr.map((v) => (Number.isFinite(Number(v)) ? Number(v) : null));
      lrSeries = { vals };
      maxLen = Math.max(maxLen, vals.length);
    }
  }

  if (!series.length || maxLen === 0) {
    setEmpty(UI.lineEmpty, true);
    if (state.charts.line) {
      state.charts.line.data.labels = [];
      state.charts.line.data.datasets = [];
      state.charts.line.update();
    }
    return;
  }

  setEmpty(UI.lineEmpty, false);

  const labels = Array.from({ length: maxLen }, (_, i) => String(i + 1));
  const datasets = series.map((s) => {
    const color = getArchColor(s.arch);
    const padded = Array.from({ length: maxLen }, (_, i) => (i < s.vals.length ? s.vals[i] : null));
    return {
      label: s.arch,
      data: padded,
      borderColor: color,
      backgroundColor: color,
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.15,
    };
  });

  const needsY2 = wantsDual && !!lrSeries;
  if (needsY2) {
    const paddedLr = Array.from({ length: maxLen }, (_, i) => (i < lrSeries.vals.length ? lrSeries.vals[i] : null));
    datasets.push({
      label: "LR",
      data: paddedLr,
      yAxisID: "y2",
      borderColor: "#111827",
      backgroundColor: "#111827",
      borderWidth: 2,
      borderDash: [6, 4],
      pointRadius: 0,
      tension: 0.15,
    });
  }

  const yTitle = metricKey === "lr_series" ? "Learning rate" : "Loss";

  if (!state.charts.line) {
    state.charts.line = new Chart(UI.lineCanvas.getContext("2d"), {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: "Step" }, grid: { display: false } },
          y: { title: { display: true, text: yTitle }, grid: { color: "#f3f4f6" } },
          ...(needsY2
            ? {
                y2: {
                  position: "right",
                  title: { display: true, text: "Learning rate" },
                  grid: { drawOnChartArea: false },
                },
              }
            : {}),
        },
      },
    });
  } else {
    state.charts.line.data.labels = labels;
    state.charts.line.data.datasets = datasets;
    state.charts.line.options.scales.y.title.text = yTitle;
    if (needsY2) {
      state.charts.line.options.scales.y2 = {
        position: "right",
        title: { display: true, text: "Learning rate" },
        grid: { drawOnChartArea: false },
      };
    } else if (state.charts.line.options.scales.y2) {
      delete state.charts.line.options.scales.y2;
    }
    state.charts.line.update();
  }
}

// ============================================================
//  DATA HELPERS FOR NEW TABS
// ============================================================

function extractNoisePct(report) {
  // From tags: "p50" -> 50, "p85" -> 85, "sig50" -> 50, "sig05" -> 95
  const tags = report.tags || [];
  for (const t of tags) {
    const pm = t.match(/^p(\d+)$/);
    if (pm) return parseInt(pm[1], 10);
    const sm = t.match(/^sig(\d+)$/);
    if (sm) return 100 - parseInt(sm[1], 10);
  }
  return null;
}

function extractVariant(report) {
  // Determine GQA vs MIG vs A-MIG from arch + tags
  const tags = report.tags || [];
  if (report.arch === "gqa") return "GQA";
  if (tags.includes("amig")) return "A-MIG";
  if (report.arch === "mig") return "MIG";
  return report.arch?.toUpperCase() || "?";
}

function groupCrossoverData(dataset) {
  const reports = getReports().filter(
    (r) => r.dataset === dataset && (r.arch === "gqa" || r.arch === "mig")
  );
  // Group by noise% then variant
  const grouped = new Map(); // noise% -> { GQA, MIG, A-MIG }
  for (const r of reports) {
    const noise = extractNoisePct(r);
    if (noise === null) continue;
    const variant = extractVariant(r);
    if (!grouped.has(noise)) grouped.set(noise, {});
    grouped.get(noise)[variant] = r;
  }
  const levels = Array.from(grouped.keys()).sort((a, b) => a - b);
  return { levels, grouped };
}

function destroyChart(key) {
  if (state.newCharts[key]) {
    state.newCharts[key].destroy();
    delete state.newCharts[key];
  }
}

function makeOrUpdateChart(key, canvas, config) {
  if (!canvas) return null;
  destroyChart(key);
  const chart = new Chart(canvas.getContext("2d"), config);
  state.newCharts[key] = chart;
  return chart;
}

// ============================================================
//  TAB 4: NOISE CROSSOVER
// ============================================================

const CROSSOVER_COLORS = { GQA: "#6366f1", MIG: "#f59e0b", "A-MIG": "#10b981" };

function selectedCrossoverDataset() {
  const active = UI.crossoverDatasetToggle?.querySelector(".seg-btn.is-active");
  return active?.dataset.value || "poisoned_needle";
}

async function renderNoiseCrossover() {
  const ds = selectedCrossoverDataset();
  const { levels, grouped } = groupCrossoverData(ds);

  if (!levels.length) {
    setEmpty(UI.crossoverEmpty, true);
    destroyChart("crossLine");
    destroyChart("crossDelta");
    if (UI.crossoverCards) UI.crossoverCards.innerHTML = "";
    return;
  }
  setEmpty(UI.crossoverEmpty, false);

  const variants = ["GQA", "MIG", "A-MIG"];

  // Need eval_loss from full reports
  const evalData = {}; // variant -> [eval_loss per level]
  for (const v of variants) evalData[v] = [];

  for (const noise of levels) {
    const row = grouped.get(noise) || {};
    for (const v of variants) {
      const r = row[v];
      if (r) {
        let ev = r.eval_loss;
        if (ev == null) {
          try {
            const full = await fetchReportJson(r.file);
            ev = full?.metrics?.eval_loss ?? null;
          } catch { ev = null; }
        }
        evalData[v].push(ev);
      } else {
        evalData[v].push(null);
      }
    }
  }

  // Line chart
  const lineDatasets = variants.map((v) => ({
    label: v,
    data: evalData[v],
    borderColor: CROSSOVER_COLORS[v],
    backgroundColor: CROSSOVER_COLORS[v],
    borderWidth: 2.5,
    pointRadius: 5,
    pointHoverRadius: 7,
    tension: 0.2,
  }));

  makeOrUpdateChart("crossLine", UI.crossoverLineCanvas, {
    type: "line",
    data: { labels: levels.map((l) => `${l}%`), datasets: lineDatasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: "top" },
        title: { display: true, text: `Eval Loss vs Noise % (${ds})` },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(4) ?? "—"}`,
          },
        },
      },
      scales: {
        x: { title: { display: true, text: "Noise %" }, grid: { display: false } },
        y: { title: { display: true, text: "Eval Loss (lower = better)" }, grid: { color: "#f3f4f6" } },
      },
    },
  });

  // Delta bar chart: (GQA - variant) per noise level
  const deltaLabels = levels.map((l) => `${l}%`);
  const deltaDatasets = ["MIG", "A-MIG"].map((v) => ({
    label: `GQA - ${v}`,
    data: levels.map((_, i) => {
      const gqa = evalData["GQA"][i];
      const val = evalData[v][i];
      if (gqa == null || val == null) return null;
      return +(gqa - val).toFixed(4);
    }),
    backgroundColor: levels.map((_, i) => {
      const gqa = evalData["GQA"][i];
      const val = evalData[v][i];
      if (gqa == null || val == null) return "#d1d5db";
      return (gqa - val) > 0 ? CROSSOVER_COLORS[v] : "#ef4444";
    }),
    borderRadius: 4,
  }));

  makeOrUpdateChart("crossDelta", UI.crossoverDeltaCanvas, {
    type: "bar",
    data: { labels: deltaLabels, datasets: deltaDatasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: "top" },
        title: { display: true, text: "Delta: GQA - Variant (positive = variant wins)" },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(4) ?? "—"}`,
          },
        },
      },
      scales: {
        x: { title: { display: true, text: "Noise %" }, grid: { display: false } },
        y: {
          title: { display: true, text: "Eval Loss Delta" },
          grid: { color: "#f3f4f6" },
        },
      },
    },
  });

  // Summary cards
  if (UI.crossoverCards) {
    let migWins = 0, amigWins = 0;
    for (let i = 0; i < levels.length; i++) {
      const g = evalData["GQA"][i];
      const m = evalData["MIG"][i];
      const a = evalData["A-MIG"][i];
      if (g != null && m != null && m < g) migWins++;
      if (g != null && a != null && a < g) amigWins++;
    }
    const total = levels.length;
    UI.crossoverCards.innerHTML = `
      <div class="summary-card">
        <span class="summary-card__label">Dataset</span>
        <span class="summary-card__value summary-card__value--neutral">${escapeHtml(ds)}</span>
        <span class="summary-card__sub">${total} noise levels</span>
      </div>
      <div class="summary-card">
        <span class="summary-card__label">MIG Wins</span>
        <span class="summary-card__value ${migWins > total / 2 ? "summary-card__value--good" : "summary-card__value--neutral"}">${migWins}/${total}</span>
        <span class="summary-card__sub">vs GQA (lower eval loss)</span>
      </div>
      <div class="summary-card">
        <span class="summary-card__label">A-MIG Wins</span>
        <span class="summary-card__value ${amigWins > total / 2 ? "summary-card__value--good" : "summary-card__value--neutral"}">${amigWins}/${total}</span>
        <span class="summary-card__sub">vs GQA (lower eval loss)</span>
      </div>
    `;
  }
}

// ============================================================
//  TAB 5: GATE DYNAMICS
// ============================================================

function selectedGateArch() {
  const active = UI.gateArchToggle?.querySelector(".seg-btn.is-active");
  return active?.dataset.value || "sil";
}

async function renderGateDynamics() {
  const gateArch = selectedGateArch();
  const reports = filteredReports().filter((r) => r.arch === gateArch);

  if (!reports.length) {
    setEmpty(UI.gatesEmpty, true);
    destroyChart("gateEvo");
    destroyChart("auxLoss");
    return;
  }

  // Fetch full reports to get series data
  const payloads = await Promise.all(
    reports.slice(0, 8).map(async (r) => {
      try {
        const json = await fetchReportJson(r.file);
        return { report: r, json };
      } catch { return null; }
    })
  );
  const valid = payloads.filter(Boolean);

  // Find reports with gate series
  const seriesKey = gateArch === "sil" ? "sil_gate_series" : "mig_gate_series";
  const withGate = valid.filter((p) => {
    const s = p.json?.metrics?.[seriesKey];
    return Array.isArray(s) && s.length > 0;
  });

  if (!withGate.length) {
    setEmpty(UI.gatesEmpty, true);
    destroyChart("gateEvo");
    destroyChart("auxLoss");
    return;
  }
  setEmpty(UI.gatesEmpty, false);

  // Gate Evolution chart
  const gateDatasets = withGate.map((p) => {
    const vals = p.json.metrics[seriesKey];
    const tags = p.report.tags || [];
    const label = `${p.report.arch} ${p.report.dataset} [${tags.join(",")}]`;
    const color = getArchColor(label);
    return {
      label,
      data: vals.map((v) => (Number.isFinite(Number(v)) ? Number(v) : null)),
      borderColor: color,
      backgroundColor: color,
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.2,
    };
  });

  const maxGateLen = Math.max(...withGate.map((p) => p.json.metrics[seriesKey].length));
  const gateLabels = Array.from({ length: maxGateLen }, (_, i) => String(i + 1));

  makeOrUpdateChart("gateEvo", UI.gateEvoCanvas, {
    type: "line",
    data: { labels: gateLabels, datasets: gateDatasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: "top", labels: { font: { size: 10 } } },
        title: { display: true, text: `${gateArch.toUpperCase()} Gate Activation Over Training` },
      },
      scales: {
        x: { title: { display: true, text: "Step" }, grid: { display: false }, ticks: { maxTicksLimit: 10 } },
        y: { title: { display: true, text: "Gate Value" }, grid: { color: "#f3f4f6" } },
      },
    },
  });

  // Aux Loss chart (total_loss - loss)
  const auxDatasets = valid
    .map((p) => {
      const loss = p.json?.metrics?.loss_series;
      const total = p.json?.metrics?.total_loss_series;
      if (!Array.isArray(loss) || !Array.isArray(total) || total.length === 0) return null;
      const aux = total.map((t, i) => {
        const l = loss[i] ?? 0;
        const tv = Number(t);
        const lv = Number(l);
        return Number.isFinite(tv) && Number.isFinite(lv) ? tv - lv : null;
      });
      const tags = p.report.tags || [];
      const label = `${p.report.arch} ${p.report.dataset} [${tags.join(",")}]`;
      const color = getArchColor(label);
      return {
        label,
        data: aux,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.2,
      };
    })
    .filter(Boolean);

  if (auxDatasets.length) {
    const maxAuxLen = Math.max(...auxDatasets.map((d) => d.data.length));
    const auxLabels = Array.from({ length: maxAuxLen }, (_, i) => String(i + 1));
    makeOrUpdateChart("auxLoss", UI.auxLossCanvas, {
      type: "line",
      data: { labels: auxLabels, datasets: auxDatasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, position: "top", labels: { font: { size: 10 } } },
          title: { display: true, text: "Auxiliary Loss (total_loss - task_loss)" },
        },
        scales: {
          x: { title: { display: true, text: "Step" }, grid: { display: false }, ticks: { maxTicksLimit: 10 } },
          y: { title: { display: true, text: "Aux Loss" }, grid: { color: "#f3f4f6" } },
        },
      },
    });
  } else {
    destroyChart("auxLoss");
  }
}

// ============================================================
//  TAB 6: TRAINING STABILITY
// ============================================================

async function renderStability(picked) {
  if (!picked.length) {
    setEmpty(UI.stabilityEmpty, true);
    ["timing", "memory", "evalGap", "paramBreakdown"].forEach((k) => destroyChart(k));
    return;
  }
  setEmpty(UI.stabilityEmpty, false);

  const archs = picked.map((p) => p.arch);
  const colors = archs.map((a) => getArchColor(a));

  // Fetch full reports for eval_loss and model breakdown
  const fullReports = await Promise.all(
    picked.map(async (p) => {
      try {
        return await fetchReportJson(p.run.file);
      } catch { return null; }
    })
  );

  // Chart A: Step Time p50 vs p95
  const p50 = picked.map((p) => toNum(p.run.train_step_ms_mean)); // mean as proxy; p50 from full
  const p95 = [];
  for (const fr of fullReports) {
    p95.push(toNum(fr?.metrics?.train_step_ms_p95));
  }

  makeOrUpdateChart("timing", UI.timingCanvas, {
    type: "bar",
    data: {
      labels: archs,
      datasets: [
        { label: "Step Mean (ms)", data: p50, backgroundColor: colors.map((c) => c.replace("45%", "60%")), borderRadius: 4 },
        { label: "Step p95 (ms)", data: p95, backgroundColor: colors, borderRadius: 4 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: "top" },
        title: { display: true, text: "Step Time: Mean vs p95" },
      },
      scales: {
        x: { grid: { display: false } },
        y: { title: { display: true, text: "ms" }, grid: { color: "#f3f4f6" } },
      },
    },
  });

  // Chart B: Peak Memory
  const mems = picked.map((p) => toNum(p.run.peak_mem_mb));
  makeOrUpdateChart("memory", UI.memoryCanvas, {
    type: "bar",
    data: {
      labels: archs,
      datasets: [{ label: "Peak Memory (MB)", data: mems, backgroundColor: colors, borderRadius: 4 }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: { display: true, text: "Peak Memory Usage" },
      },
      scales: {
        x: { title: { display: true, text: "MB" }, grid: { color: "#f3f4f6" } },
        y: { grid: { display: false } },
      },
    },
  });

  // Chart C: Eval vs Train Loss Gap
  const trainLoss = picked.map((p) => toNum(p.run.loss_mean));
  const evalLoss = fullReports.map((fr) => toNum(fr?.metrics?.eval_loss));
  makeOrUpdateChart("evalGap", UI.evalGapCanvas, {
    type: "bar",
    data: {
      labels: archs,
      datasets: [
        { label: "Train Loss", data: trainLoss, backgroundColor: "rgba(99, 102, 241, 0.5)", borderRadius: 4 },
        { label: "Eval Loss", data: evalLoss, backgroundColor: "rgba(239, 68, 68, 0.6)", borderRadius: 4 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: "top" },
        title: { display: true, text: "Train vs Eval Loss (gap = generalization)" },
      },
      scales: {
        x: { grid: { display: false } },
        y: { title: { display: true, text: "Loss" }, grid: { color: "#f3f4f6" } },
      },
    },
  });

  // Chart D: Parameter Breakdown
  const embed = [], mixin = [], ffn = [], head = [];
  for (const fr of fullReports) {
    const m = fr?.model || {};
    embed.push(toNum(m.embedding_parameters));
    mixin.push(toNum(m.mixin_parameters));
    ffn.push(toNum(m.ffn_parameters));
    head.push(toNum(m.lm_head_parameters));
  }
  makeOrUpdateChart("paramBreakdown", UI.paramBreakdownCanvas, {
    type: "bar",
    data: {
      labels: archs,
      datasets: [
        { label: "Embedding", data: embed, backgroundColor: "#818cf8", borderRadius: 2 },
        { label: "Mixin (attn)", data: mixin, backgroundColor: "#f59e0b", borderRadius: 2 },
        { label: "FFN", data: ffn, backgroundColor: "#10b981", borderRadius: 2 },
        { label: "LM Head", data: head, backgroundColor: "#6b7280", borderRadius: 2 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: "top" },
        title: { display: true, text: "Parameter Breakdown" },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const v = ctx.parsed.y;
              return `${ctx.dataset.label}: ${v != null ? (v / 1e6).toFixed(3) + "M" : "—"}`;
            },
          },
        },
      },
      scales: {
        x: { stacked: true, grid: { display: false } },
        y: { stacked: true, title: { display: true, text: "Parameters" }, grid: { color: "#f3f4f6" } },
      },
    },
  });
}

// ============================================================
//  TAB 7: CROSS-TRACK RADAR
// ============================================================

const RADAR_AXES = [
  { id: "quality", label: "Quality", extract: (p, fr) => { const e = toNum(fr?.metrics?.eval_loss ?? p.run.eval_loss); return e ? 1 / e : null; }, better: "higher" },
  { id: "throughput", label: "Throughput", extract: (p) => toNum(p.run.tokens_per_s), better: "higher" },
  { id: "memory", label: "Memory Eff.", extract: (p) => { const m = toNum(p.run.peak_mem_mb); return m ? 1 / m : null; }, better: "higher" },
  { id: "paramEff", label: "Tok/s per MP", extract: (p) => toNum(p.run.tokens_per_s_per_mparam), better: "higher" },
  { id: "speed", label: "Train Speed", extract: (p) => { const s = toNum(p.run.train_step_ms_mean); return s ? 1 / s : null; }, better: "higher" },
  { id: "generalization", label: "Generalization", extract: (p, fr) => { const ev = toNum(fr?.metrics?.eval_loss ?? p.run.eval_loss); const tr = toNum(p.run.loss_mean); return (ev != null && tr != null && ev > tr) ? 1 / (ev - tr) : null; }, better: "higher" },
];

async function renderRadar(picked) {
  if (picked.length < 2) {
    setEmpty(UI.radarEmpty, true);
    destroyChart("radar");
    return;
  }
  setEmpty(UI.radarEmpty, false);

  const fullReports = await Promise.all(
    picked.map(async (p) => {
      try { return await fetchReportJson(p.run.file); } catch { return null; }
    })
  );

  // Extract raw values per axis per arch
  const raw = RADAR_AXES.map((axis) =>
    picked.map((p, i) => axis.extract(p, fullReports[i]))
  );

  // Min-max normalize each axis to [0, 1]
  const norm = raw.map((axisVals) => {
    const nums = axisVals.filter((v) => v != null);
    if (nums.length === 0) return axisVals.map(() => 0);
    const min = Math.min(...nums);
    const max = Math.max(...nums);
    const range = max - min || 1;
    return axisVals.map((v) => (v != null ? (v - min) / range : 0));
  });

  const datasets = picked.map((p, pi) => {
    const color = getArchColor(p.arch);
    return {
      label: p.arch,
      data: RADAR_AXES.map((_, ai) => norm[ai][pi]),
      borderColor: color,
      backgroundColor: color.replace("45%)", "45%, 0.15)").replace("hsl", "hsla"),
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
    };
  });

  makeOrUpdateChart("radar", UI.radarCanvas, {
    type: "radar",
    data: { labels: RADAR_AXES.map((a) => a.label), datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: "top" },
        title: { display: true, text: "Cross-Track Architecture Comparison (normalized)" },
      },
      scales: {
        r: {
          beginAtZero: true,
          max: 1,
          ticks: { display: false },
          grid: { color: "#e5e7eb" },
          pointLabels: { font: { size: 12, weight: "600" } },
        },
      },
    },
  });

  // "Best at" summary table
  const thead = UI.radarBestTable?.querySelector("thead");
  const tbody = UI.radarBestTable?.querySelector("tbody");
  if (thead && tbody) {
    thead.innerHTML = "<tr><th>Axis</th><th>Best Architecture</th><th>Value</th></tr>";
    tbody.innerHTML = "";
    for (let ai = 0; ai < RADAR_AXES.length; ai++) {
      const axis = RADAR_AXES[ai];
      let bestVal = -Infinity;
      let bestArch = "—";
      let bestRaw = null;
      for (let pi = 0; pi < picked.length; pi++) {
        const v = raw[ai][pi];
        if (v != null && v > bestVal) {
          bestVal = v;
          bestArch = picked[pi].arch;
          bestRaw = v;
        }
      }
      const row = document.createElement("tr");
      const displayVal = bestRaw != null ? bestRaw.toFixed(4) : "—";
      row.innerHTML = `<td>${escapeHtml(axis.label)}</td><td style="color:${getArchColor(bestArch)}; font-weight:720">${escapeHtml(bestArch)}</td><td class="val">${displayVal}</td>`;
      tbody.appendChild(row);
    }
  }
}

// ============================================================
//  TAB 8: RUN INSPECTOR
// ============================================================

function fillInspectorSelect() {
  const reports = filteredReports();
  const prev = UI.inspectorRunSelect?.value;
  if (!UI.inspectorRunSelect) return;
  UI.inspectorRunSelect.innerHTML = "";

  const opt0 = document.createElement("option");
  opt0.value = "";
  opt0.textContent = "-- Select a run --";
  UI.inspectorRunSelect.appendChild(opt0);

  for (const r of reports) {
    const o = document.createElement("option");
    o.value = r.file;
    const ts = (r.timestamp_utc || "").slice(0, 19).replace("T", " ");
    o.textContent = `${r.arch} / ${r.dataset} / ${ts}`;
    UI.inspectorRunSelect.appendChild(o);
  }
  if (prev) UI.inspectorRunSelect.value = prev;
}

async function renderInspector() {
  const file = UI.inspectorRunSelect?.value;
  if (!file) {
    setEmpty(UI.inspectorEmpty, true);
    destroyChart("inspLoss");
    destroyChart("inspGate");
    if (UI.inspectorCards) UI.inspectorCards.innerHTML = "";
    if (UI.inspectorConfigTable) UI.inspectorConfigTable.querySelector("tbody").innerHTML = "";
    return;
  }
  setEmpty(UI.inspectorEmpty, false);

  let json;
  try {
    json = await fetchReportJson(file);
  } catch (e) {
    setEmpty(UI.inspectorEmpty, true);
    return;
  }

  const run = json.run || {};
  const cfg = json.config || {};
  const met = json.metrics || {};
  const sys = json.system || {};
  const model = json.model || {};

  // Info cards
  if (UI.inspectorCards) {
    const tags = (run.tags || []).map((t) => `<span class="tag-badge">${escapeHtml(t)}</span>`).join("");
    UI.inspectorCards.innerHTML = `
      <div class="info-card"><div class="info-card__label">Run ID</div><div class="info-card__value">${escapeHtml(run.id || "—")}</div></div>
      <div class="info-card"><div class="info-card__label">Timestamp</div><div class="info-card__value">${escapeHtml((run.timestamp_utc || "").slice(0, 19))}</div></div>
      <div class="info-card"><div class="info-card__label">Architecture</div><div class="info-card__value" style="color:${getArchColor(cfg.arch || "")}">${escapeHtml(cfg.arch || "—")}</div></div>
      <div class="info-card"><div class="info-card__label">Dataset</div><div class="info-card__value">${escapeHtml(cfg.dataset || "—")}</div></div>
      <div class="info-card"><div class="info-card__label">Device</div><div class="info-card__value">${escapeHtml(cfg.device || "—")}</div></div>
      <div class="info-card"><div class="info-card__label">Eval Loss</div><div class="info-card__value">${met.eval_loss != null ? Number(met.eval_loss).toFixed(4) : "—"}</div></div>
      <div class="info-card"><div class="info-card__label">Params</div><div class="info-card__value">${model.total_parameters ? (model.total_parameters / 1e6).toFixed(2) + "M" : "—"}</div></div>
      <div class="info-card"><div class="info-card__label">GPU</div><div class="info-card__value">${escapeHtml(sys.gpu_name || "—")}</div></div>
      ${tags ? `<div class="info-card" style="flex: 2 1 300px"><div class="info-card__label">Tags</div><div class="info-card__value">${tags}</div></div>` : ""}
    `;
  }

  // Loss curve
  const lossSeries = met.loss_series;
  const totalLoss = met.total_loss_series;
  if (Array.isArray(lossSeries) && lossSeries.length > 0) {
    const labels = Array.from({ length: lossSeries.length }, (_, i) => String(i + 1));
    const datasets = [
      {
        label: "Task Loss",
        data: lossSeries.map((v) => (Number.isFinite(Number(v)) ? Number(v) : null)),
        borderColor: "#6366f1",
        backgroundColor: "#6366f1",
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.15,
      },
    ];
    if (Array.isArray(totalLoss) && totalLoss.length > 0) {
      datasets.push({
        label: "Total Loss (incl. aux)",
        data: totalLoss.map((v) => (Number.isFinite(Number(v)) ? Number(v) : null)),
        borderColor: "#f59e0b",
        backgroundColor: "#f59e0b",
        borderWidth: 1.5,
        borderDash: [4, 3],
        pointRadius: 0,
        tension: 0.15,
      });
    }
    makeOrUpdateChart("inspLoss", UI.inspectorLossCanvas, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, position: "top" },
          title: { display: true, text: "Loss Curve" },
        },
        scales: {
          x: { title: { display: true, text: "Step" }, grid: { display: false }, ticks: { maxTicksLimit: 10 } },
          y: { title: { display: true, text: "Loss" }, grid: { color: "#f3f4f6" } },
        },
      },
    });
  } else {
    destroyChart("inspLoss");
  }

  // Gate series (if any)
  const gateSeries = met.mig_gate_series || met.sil_gate_series;
  if (Array.isArray(gateSeries) && gateSeries.length > 0) {
    const labels = Array.from({ length: gateSeries.length }, (_, i) => String(i + 1));
    makeOrUpdateChart("inspGate", UI.inspectorGateCanvas, {
      type: "line",
      data: {
        labels,
        datasets: [{
          label: met.mig_gate_series ? "MIG Gate" : "SIL Gate",
          data: gateSeries.map((v) => (Number.isFinite(Number(v)) ? Number(v) : null)),
          borderColor: "#10b981",
          backgroundColor: "#10b981",
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.15,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          title: { display: true, text: "Gate Activation Series" },
        },
        scales: {
          x: { title: { display: true, text: "Step" }, grid: { display: false }, ticks: { maxTicksLimit: 10 } },
          y: { title: { display: true, text: "Gate Value" }, grid: { color: "#f3f4f6" } },
        },
      },
    });
  } else {
    destroyChart("inspGate");
  }

  // Config table
  const tbody = UI.inspectorConfigTable?.querySelector("tbody");
  if (tbody) {
    tbody.innerHTML = "";
    const entries = Object.entries(cfg).sort(([a], [b]) => a.localeCompare(b));
    for (const [k, v] of entries) {
      const row = document.createElement("tr");
      row.innerHTML = `<td>${escapeHtml(k)}</td><td class="val">${escapeHtml(String(v ?? "—"))}</td>`;
      tbody.appendChild(row);
    }
    // Add system info rows
    const sysEntries = [
      ["system.os", sys.os],
      ["system.python", sys.python],
      ["system.torch", sys.torch],
      ["system.gpu", sys.gpu_name],
      ["system.cpu", sys.cpu_brand],
      ["system.ram_gb", sys.ram_gb],
    ];
    for (const [k, v] of sysEntries) {
      if (v == null) continue;
      const row = document.createElement("tr");
      row.innerHTML = `<td>${escapeHtml(k)}</td><td class="val">${escapeHtml(String(v))}</td>`;
      tbody.appendChild(row);
    }
  }
}

// ============================================================

function updateSubtitle() {
  const ds = selectedDataset();
  const dev = selectedDevice();
  const mode = UI.aggMode.value;
  const baseline = UI.baselineArch.value;
  const modeTxt = mode === "latest" ? "Latest" : "Best (min loss)";

  const total = getReports().length;
  const shown = filteredReports().length;
  const countTxt = total > 0 ? ` • Reports: ${shown}/${total}` : "";
  const devTxt = dev && dev !== "all" ? ` • Device: ${dev.toUpperCase()}` : "";
  UI.tableSubtitle.textContent = `Dataset: ${ds}${devTxt}${countTxt} • Mode: ${modeTxt}${baseline ? ` • Baseline: ${baseline}` : ""}`;
}

async function updateAll() {
  const picked = getPicked();
  updateSubtitle();
  renderTable(picked);
  renderScatter(picked);
  await renderLine(picked);

  // Re-render active new-tab (if any)
  const activeTab = document.querySelector(".tab.is-active")?.id;
  if (activeTab === "tab-crossover") await renderNoiseCrossover();
  if (activeTab === "tab-gates") await renderGateDynamics();
  if (activeTab === "tab-stability") await renderStability(picked);
  if (activeTab === "tab-radar") await renderRadar(picked);
  if (activeTab === "tab-inspector") { fillInspectorSelect(); await renderInspector(); }
}

function wireEvents() {
  UI.refresh?.addEventListener("click", async () => {
    try {
      await loadManifest();
      fillDatasetSelect();
      renderArchList();
      fillSeriesSelect();
      await updateAll();
    } catch (e) {
      console.error(e);
      alert(String(e));
    }
  });

  UI.dataset?.addEventListener("change", () => {
    renderArchList();
    updateAll().catch((e) => alert(String(e)));
  });

  // Device toggle (segmented buttons)
  if (UI.deviceToggle) {
    for (const btn of UI.deviceToggle.querySelectorAll(".seg-btn")) {
      btn.addEventListener("click", () => {
        UI.deviceToggle.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        renderArchList();
        updateAll().catch((e) => alert(String(e)));
      });
    }
  }

  UI.aggMode?.addEventListener("change", () => {
    updateAll().catch((e) => alert(String(e)));
  });

  UI.baselineArch?.addEventListener("change", () => {
    updateAll().catch((e) => alert(String(e)));
  });

  UI.seriesMetric?.addEventListener("change", () => {
    renderLine(getPicked()).catch((e) => alert(String(e)));
  });

  UI.archList?.addEventListener("change", () => {
    updateAll().catch((e) => alert(String(e)));
  });

  UI.archAll?.addEventListener("click", () => {
    UI.archList?.querySelectorAll("input[type=checkbox]").forEach((n) => (n.checked = true));
    updateAll().catch((e) => alert(String(e)));
  });

  UI.archNone?.addEventListener("click", () => {
    UI.archList?.querySelectorAll("input[type=checkbox]").forEach((n) => (n.checked = false));
    updateAll().catch((e) => alert(String(e)));
  });

  UI.sidebarToggle?.addEventListener("click", () => {
    if (!UI.sidebar) return;
    const collapsed = UI.sidebar.classList.toggle("sidebar--collapsed");
    SHELL?.classList.toggle("app-shell--sidebar-collapsed", collapsed);
  });

  // New tab controls
  if (UI.crossoverDatasetToggle) {
    for (const btn of UI.crossoverDatasetToggle.querySelectorAll(".seg-btn")) {
      btn.addEventListener("click", () => {
        UI.crossoverDatasetToggle.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        renderNoiseCrossover().catch(console.error);
      });
    }
  }
  if (UI.gateArchToggle) {
    for (const btn of UI.gateArchToggle.querySelectorAll(".seg-btn")) {
      btn.addEventListener("click", () => {
        UI.gateArchToggle.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        renderGateDynamics().catch(console.error);
      });
    }
  }
  UI.inspectorRunSelect?.addEventListener("change", () => {
    renderInspector().catch(console.error);
  });
}

async function main() {
  wireTabs();
  await loadManifest();
  fillDatasetSelect();
  fillSeriesSelect();
  renderArchList();
  fillInspectorSelect();
  wireEvents();
  await updateAll();

  // One extra resize pass in case charts were initialized hidden.
  const activeTab = document.querySelector(".tab.is-active")?.id;
  if (activeTab) activateTabById(activeTab);
}

main().catch((e) => {
  console.error(e);
  alert(String(e));
});
