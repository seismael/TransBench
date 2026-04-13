// TransBench Analytics: Comparative Analysis Dashboard
// - Efficiency Frontier scatter (loss vs tokens/s)
// - Dense pivot table with heatmap + optional baseline deltas
// - Convergence chart (loss/lr series) over steps

const MANIFEST_URL = "/reports/manifest.json";

const SHELL = document.querySelector(".app-shell");

const UI = {
  refresh: document.getElementById("refresh"),
  dataset: document.getElementById("dataset"),
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
};

const state = {
  manifest: null,
  charts: { scatter: null, line: null },
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
const ALLOWED_DATASETS = ["tinystories", "synthetic", "zeros", "ramp", "tinystories-instruct", "wikitext"];

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

function filteredReports() {
  const reports = getReports();
  const ds = selectedDataset();
  if (!ds || ds === "all") return reports;
  return reports.filter((r) => (r.dataset || "unknown") === ds);
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
  const res = await fetch(`/reports/${encodeURIComponent(file)}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch report: ${file}`);
  return await res.json();
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

function updateSubtitle() {
  const ds = selectedDataset();
  const mode = UI.aggMode.value;
  const baseline = UI.baselineArch.value;
  const modeTxt = mode === "latest" ? "Latest" : "Best (min loss)";

  const total = getReports().length;
  const shown = filteredReports().length;
  const countTxt = total > 0 ? ` • Reports: ${shown}/${total}` : "";
  UI.tableSubtitle.textContent = `Dataset: ${ds}${countTxt} • Mode: ${modeTxt}${baseline ? ` • Baseline: ${baseline}` : ""}`;
}

async function updateAll() {
  const picked = getPicked();
  updateSubtitle();
  renderTable(picked);
  renderScatter(picked);
  await renderLine(picked);
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
}

async function main() {
  wireTabs();
  await loadManifest();
  fillDatasetSelect();
  fillSeriesSelect();
  renderArchList();
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
