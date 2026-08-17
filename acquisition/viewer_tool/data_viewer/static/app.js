const state = {
  index: null,
  samples: [],
  selectedId: null,
  selectionToken: 0,
  playbackMode: "mano",
};

const els = {
  title: document.querySelector("#app-title"),
  status: document.querySelector("#status"),
  search: document.querySelector("#search-input"),
  folderFilter: document.querySelector("#folder-filter"),
  objectFilter: document.querySelector("#object-filter"),
  count: document.querySelector("#sample-count"),
  list: document.querySelector("#sample-list"),
  empty: document.querySelector("#empty-state"),
  detail: document.querySelector("#detail"),
  sampleTitle: document.querySelector("#sample-title"),
  samplePath: document.querySelector("#sample-path"),
  viser: document.querySelector("#viser-frame"),
  metadata: document.querySelector("#metadata"),
  modeButtons: [...document.querySelectorAll("[data-playback-mode]")],
};

els.search.addEventListener("input", renderSampleList);
for (const filter of [els.folderFilter, els.objectFilter]) {
  filter.addEventListener("change", () => {
    renderSampleList();
    selectFirstVisibleSample();
  });
}
for (const button of els.modeButtons) {
  button.addEventListener("click", () => setPlaybackMode(button.dataset.playbackMode));
}
loadProject();

async function loadProject() {
  setStatus("Indexing sessions...");
  try {
    const response = await fetch("/api/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    if (payload.error) {
      setStatus(payload.message || payload.error);
      return;
    }
    state.index = payload.index;
    state.samples = payload.index.samples || [];
    els.title.textContent = payload.index.title || "Motion Archive";
    setStatus(`${payload.index.sample_count} sessions · ${basename(payload.index.root)}`);
    populateFolderFilter();
    populateObjectFilter();
    renderSampleList();
    if (state.samples.length) selectSample(state.samples[0].id);
  } catch (error) {
    setStatus(`Unable to load project · ${String(error)}`);
  }
}

function renderSampleList() {
  const query = els.search.value.trim().toLowerCase();
  const selectedFolder = els.folderFilter.value;
  const selectedObject = els.objectFilter.value;
  const samples = state.samples.filter((sample) => {
    if (selectedFolder && sampleFolder(sample) !== selectedFolder) return false;
    if (selectedObject && sample.facets?.object !== selectedObject) return false;
    if (!query) return true;
    const haystack = [
      sample.label,
      sample.summary,
      sample.path,
      sample.group_label,
      ...Object.values(sample.facets || {}),
    ].join(" ").toLowerCase();
    return haystack.includes(query);
  });
  els.count.textContent = `${samples.length} / ${state.samples.length} sessions`;
  els.list.innerHTML = "";
  let rendered = 0;
  const grouped = groupSamplesByDataset(samples);
  for (const [dataset, datasetSamples] of grouped) {
    if (rendered >= 1000) break;
    const datasetHeader = document.createElement("div");
    datasetHeader.className = "dataset-group";
    datasetHeader.textContent = dataset;
    els.list.append(datasetHeader);
    for (const sample of datasetSamples) {
      if (rendered >= 1000) break;
      els.list.append(createSampleButton(sample));
      rendered += 1;
    }
  }
}

function populateFolderFilter() {
  const folders = [...new Set(state.samples.map((sample) => sampleFolder(sample)).filter(Boolean))].sort();
  const previous = els.folderFilter.value;
  els.folderFilter.innerHTML = `<option value="">All folders</option>`;
  for (const folder of folders) {
    const option = document.createElement("option");
    option.value = folder;
    option.textContent = folder;
    els.folderFilter.append(option);
  }
  if (previous && folders.includes(previous)) els.folderFilter.value = previous;
}

function sampleFolder(sample) {
  return sample.facets?.dataset || sample.group_label || "";
}

function populateObjectFilter() {
  const objects = [...new Set(state.samples.map((sample) => sample.facets?.object).filter(Boolean))].sort();
  els.objectFilter.innerHTML = `<option value="">All objects</option>`;
  for (const object of objects) {
    const option = document.createElement("option");
    option.value = object;
    option.textContent = object;
    els.objectFilter.append(option);
  }
}

function selectFirstVisibleSample() {
  const selectedFolder = els.folderFilter.value;
  const selectedObject = els.objectFilter.value;
  const first = state.samples.find((sample) =>
    (!selectedFolder || sampleFolder(sample) === selectedFolder) &&
    (!selectedObject || sample.facets?.object === selectedObject)
  );
  if (first) selectSample(first.id);
}

function groupSamplesByDataset(samples) {
  const datasets = new Map();
  for (const sample of samples) {
    const dataset = sample.facets?.dataset || sample.group_label || "default";
    if (!datasets.has(dataset)) datasets.set(dataset, []);
    datasets.get(dataset).push(sample);
  }
  return datasets;
}

function createSampleButton(sample) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `sample-item${sample.id === state.selectedId ? " active" : ""}`;
  button.setAttribute("aria-pressed", sample.id === state.selectedId ? "true" : "false");
  button.innerHTML = `
    <span class="sample-main">${escapeHtml(sample.label)}</span>
    ${sample.summary ? `<span class="sample-sub">${escapeHtml(sample.summary)}</span>` : ""}
  `;
  button.addEventListener("click", () => selectSample(sample.id));
  return button;
}

async function selectSample(id) {
  state.selectedId = id;
  state.selectionToken += 1;
  const token = state.selectionToken;
  renderSampleList();
  const sample = state.samples.find((item) => item.id === id);
  if (!sample) return;
  els.empty.classList.add("hidden");
  els.detail.classList.remove("hidden");
  els.sampleTitle.textContent = sampleTitle(sample);
  els.samplePath.textContent = sample.path;
  renderMetadataMessage("Loading metadata...");
  els.viser.innerHTML = `<div class="status-note">Preparing 3D scene...</div>`;

  const detail = await fetchJson(`/api/sample?id=${encodeURIComponent(id)}`);
  if (token !== state.selectionToken) return;
  if (detail.error) {
    renderMetadataMessage(detail.message || detail.error);
  } else {
    renderMetadataGrid(detail.metadata || {});
  }
  await loadViser(id, token);
}

function setPlaybackMode(mode) {
  if (!mode || mode === state.playbackMode) return;
  state.playbackMode = mode;
  renderPlaybackMode();
  if (!state.selectedId) return;
  state.selectionToken += 1;
  const token = state.selectionToken;
  els.viser.innerHTML = `<div class="status-note">Preparing 3D scene...</div>`;
  loadViser(state.selectedId, token);
}

function renderPlaybackMode() {
  for (const button of els.modeButtons) {
    const active = button.dataset.playbackMode === state.playbackMode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  }
}


function renderMetadataMessage(message) {
  els.metadata.classList.add("metadata-message");
  els.metadata.textContent = message;
}

function renderMetadataGrid(metadata) {
  const rows = metadataRows(metadata);
  if (!rows.length) {
    renderMetadataMessage("No metadata.");
    return;
  }
  els.metadata.classList.remove("metadata-message");
  els.metadata.innerHTML = rows.map((row) => `
    <div class="metadata-row">
      <div class="metadata-key">${escapeHtml(metadataKey(row))}</div>
      <div class="metadata-value">${escapeHtml(formatMetadataValue(row.value))}</div>
    </div>
  `).join("");
}

function metadataKey(row) {
  return row.key;
}

function metadataRows(metadata) {
  const record = metadata.record || {};
  const fields = metadata.fields || {};
  const assets = metadata.assets || {};
  const build = metadata.build_options || {};
  const rows = [];

  addRows(rows, "Overview", [
    ["Object", record.object_name],
    ["Task", record.task_name],
    ["Motion", record.motion_name || basename(record.path)],
    ["Subject", record.subject_id],
    ["FPS", record.fps],
    ["Frames", frameSummary(record)],
    ["Source", record.source_path],
    ["File", record.path],
  ]);

  addRows(rows, "Processing", [
    ["Trim", trimSummary(record)],
    ["Contact frames", record.contact_frames],
    ["Original frames", record.n_frames_original],
    ["Sampled frames", record.n_frames_sampled || record.n_frames],
  ]);

  addRows(rows, "Playback", [
    ["Stride", build.stride],
    ["Max frames", build.max_frames],
    ["Object visible", build.show_object],
    ["Object scale", build.object_scale],
  ]);

  addRows(rows, "Assets", [
    ["Object mesh", compactPath(assets.object_mesh)],
    ["MANO left", compactPath(assets.mano_left)],
    ["MANO right", compactPath(assets.mano_right)],
  ]);

  for (const [name, info] of Object.entries(fields)) {
    rows.push({ section: "Fields", key: name, value: fieldSummary(info) });
  }

  addRows(rows, "MANO shape (β)", [
    ["Available", metadata.mano?.available ? "yes" : "no"],
  ]);
  for (const [side, info] of Object.entries(metadata.mano?.sides || {})) {
    if (!info.available) continue;
    const beta = (info.beta || []).map((v) => Number(v).toFixed(2)).join(", ");
    rows.push({ section: "MANO shape (β)", key: `${side} β`, value: `[${beta}]` });
  }

  const known = new Set(["record", "fields", "assets", "build_options", "mano"]);
  for (const [key, value] of Object.entries(metadata)) {
    if (known.has(key)) continue;
    for (const [path, item] of flattenMetadata(value, key)) {
      rows.push({ section: "Extra", key: path, value: item });
    }
  }
  return rows;
}

function addRows(rows, section, pairs) {
  for (const [key, value] of pairs) {
    if (value === undefined || value === null || value === "") continue;
    rows.push({ section, key, value });
  }
}

function frameSummary(record) {
  const sampled = record.n_frames_sampled || record.n_frames;
  const original = record.n_frames_original;
  if (sampled && original && sampled !== original) return `${sampled} / ${original}`;
  return sampled || original;
}

function trimSummary(record) {
  const method = record.trim_method || "unknown";
  const threshold = record.trim_distance_threshold_m;
  if (threshold === undefined || threshold === null) return method;
  return `${method} @ ${threshold} m`;
}

function fieldSummary(info) {
  if (!isPlainObject(info)) return info;
  const shape = Array.isArray(info.shape) ? `[${info.shape.join(", ")}]` : "[]";
  return `${info.dtype || "unknown"} ${shape}`;
}

function compactPath(path) {
  if (!path) return path;
  const text = String(path);
  const marker = "/do-as-i-do/";
  const index = text.indexOf(marker);
  if (index >= 0) return text.slice(index + marker.length);
  return text;
}

function basename(path) {
  if (!path) return path;
  return String(path).split("/").pop();
}

function flattenMetadata(value, prefix = "") {
  if (!isPlainObject(value)) return prefix ? [[prefix, value]] : [];
  const rows = [];
  for (const [key, item] of Object.entries(value)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (isPlainObject(item)) {
      const childRows = flattenMetadata(item, path);
      if (childRows.length) rows.push(...childRows);
      else rows.push([path, "{}"]);
    } else {
      rows.push([path, item]);
    }
  }
  return rows;
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function formatMetadataValue(value) {
  if (value === null) return "null";
  if (value === undefined) return "undefined";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value, null, 2);
}

async function loadViser(id, token) {
  const data = await fetchJson(
    `/api/viser?id=${encodeURIComponent(id)}&mode=${encodeURIComponent(state.playbackMode)}`
  );
  if (token !== state.selectionToken) return;
  if (data.error) {
    els.viser.innerHTML = `<div class="status-note">Playback unavailable: ${escapeHtml(data.message || data.error)}</div>`;
    return;
  }
  const src = `${data.client}?playbackPath=${encodeURIComponent(data.path)}`;
  els.viser.innerHTML = `
    <iframe class="viser-iframe" src="${escapeHtml(src)}" allow="fullscreen"></iframe>
    <div class="file-meta">${escapeHtml(data.label || data.name)}</div>
  `;
}

async function fetchJson(url) {
  try {
    const response = await fetch(url);
    return await response.json();
  } catch (error) {
    return { error: "request_failed", message: String(error) };
  }
}

function setStatus(message) {
  els.status.textContent = message;
}

function sampleTitle(sample) {
  const parts = [
    sample.facets?.dataset,
    sample.facets?.object,
    sample.label,
  ].filter(Boolean);
  return parts.length ? parts.join(" / ") : sample.label;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
