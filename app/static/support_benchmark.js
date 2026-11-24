import { bindManual } from "./support_manual.js";
import { mountLlmConsole } from "./support_llm.js";

const pretty = value => JSON.stringify(value, null, 2);
let currentBenchmarkJobId = null;
let benchmarkPollTimer = null;
let currentKbJobId = null;
let kbPollTimer = null;

async function parseJsonResponse(response, options = {}) {
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = {
      status: "error",
      message: text || response.statusText || "Non-JSON server response"
    };
  }
  if (!response.ok || (!options.allowApplicationError && data.status === "error")) {
    throw new Error(pretty(data));
  }
  return data;
}

async function jsonRequest(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  return parseJsonResponse(response);
}

function show(id, text) {
  const node = document.getElementById(id);
  node.textContent = text;
  node.style.display = "block";
}

window.loadOptions = async function loadOptions() {
  const data = await (await fetch("/benchmark-workbench/options")).json();
  datasets.textContent = data.benchmark_datasets.length
    ? "Existing JSONL benchmarks:\n" + data.benchmark_datasets
        .map(item => `- ${item.name} (${Math.round((item.size || 0) / 1024)} KB)`)
        .join("\n")
    : "No JSONL benchmark dataset listed yet.";
};

window.loadKbStatus = async function loadKbStatus() {
  try {
    const data = await (await fetch("/benchmark-workbench/kb/status")).json();
    kbOutput.textContent = pretty(data);
    const activeProcessedCount = data?.active_kb?.processed_chunk_count;
    const activeGraphStatus = data?.active_kb?.keyword_graph_exists ? "found" : "not found";
    const buildCount = Array.isArray(data?.staged_builds) ? data.staged_builds.length : 0;
    const latest = data?.latest_successful_staged_build || data?.latest_staged_build || null;
    const latestBuildId = latest?.build_id || "none";
    const latestStatus = latest?.status || "not available";
    const latestChunks = latest?.ingest?.chunks ?? latest?.ingest?.documents ?? "n/a";
    show(
      "kbStatus",
      `Active KB chunks JSONL: ${activeProcessedCount ?? "not found"}; active keyword graph: ${activeGraphStatus}; latest staged build: ${latestBuildId} (${latestStatus}, chunks=${latestChunks}); staged builds listed: ${buildCount}.`
    );
  } catch (error) {
    kbOutput.textContent = String(error);
    show("kbStatus", "KB status request failed.");
  }
};

window.chooseKbDirectory = function chooseKbDirectory() {
  kbDirPicker.click();
};

window.usePickedKbDirectory = function usePickedKbDirectory(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const nativePath = file.path || "";
  if (nativePath) {
    const normalized = nativePath.replace(/\\/g, "/");
    const folderPath = normalized.replace(/\/[^/]+$/, "").replace(/\//g, "\\");
    if (folderPath) {
      kbSourceDir.value = folderPath;
      show("kbStatus", `Selected source directory: ${folderPath}`);
      return;
    }
  }
  show("kbStatus", "The browser did not expose an absolute folder path. Paste the absolute directory path manually before starting the staged KB build.");
};

function latestKbLogLine(data) {
  const stderrTail = String(data?.stderr_tail || "").trim().split(/\r?\n/).filter(Boolean);
  const stdoutTail = String(data?.stdout_tail || "").trim().split(/\r?\n/).filter(Boolean);
  return stderrTail.at(-1) || stdoutTail.at(-1) || "";
}

window.startKbConstruction = async function startKbConstruction() {
  show("kbStatus", "Starting staged KB construction job.");
  try {
    const data = await jsonRequest("/benchmark-workbench/kb/start", {
      source_dir: kbSourceDir.value,
      build_name: kbBuildName.value || "kb.paper.staged",
      pdf_parser_backend: kbPdfParserBackend.value,
      chunk_mode: kbChunkMode.value,
      chunk_token_size: Number(kbChunkTokenSize.value),
      chunk_sentence_overlap: Number(kbSentenceOverlap.value),
      label_mode: "none",
      build_index: kbBuildIndex.checked,
      build_keyword_graph: kbBuildKeywordGraph.checked,
      embedding_model: kbEmbeddingModel.value || null,
      embedding_local_files_only: kbEmbeddingLocal.checked,
      allow_hash_fallback: kbAllowHashFallback.checked
    });
    currentKbJobId = data.job_id;
    kbBuildBtn.disabled = true;
    kbOutput.textContent = pretty(data);
    show("kbStatus", `Staged KB job started: ${currentKbJobId}`);
    pollKbConstructionJob();
  } catch (error) {
    kbOutput.textContent = String(error);
    show("kbStatus", "Staged KB construction request failed.");
  }
};

window.showChooseFolderHint = function showChooseFolderHint() {
  window.alert(`Choose folder can only fill a usable path when the browser exposes a native file path. If the browser returns only a folder label, paste the full directory path of your PDF folder manually.`);
};

window.showBuildGroupHint = function showBuildGroupHint() {
  window.alert(`Build staged KB parses the selected PDF directory and writes a separate staged KB under <project_root>/outputs/kb_construction/<build_id>/. Build staged KB + graph runs the same staged KB construction path and then builds the staged keyword graph.\n\nCurrent keyword graph defaults are:\n- min_doc_freq = 4\n- max_doc_freq_ratio = 0.18\n- min_edge_weight = 3\n\nThese thresholds were tuned for larger corpora. Very small staged tests, such as one or two papers, can therefore produce zero retained keyword nodes and edges even when PDF parsing and chunk construction succeed.`);
};

window.showActivateGroupHint = function showActivateGroupHint() {
  window.alert(`Activate staged KB replaces the current active KB with the latest successful staged build. Merge staged KB into active KB appends the latest successful staged build into the current active KB, then rebuilds the active embedding index and active keyword graph. After either route, the resulting active KB can be used directly by downstream RAG routes and by QA benchmark construction from the active KB chunks JSONL.`);
};

window.showInspectGroupHint = function showInspectGroupHint() {
  window.alert(`Open active keyword graph shows the keyword graph currently stored under petroleum_kb/kb_core/data/analysis. Open latest staged keyword graph shows the newest successful staged graph from the staged build directory without activating it as the production KB.`);
};

window.startKbConstructionWithGraph = function startKbConstructionWithGraph() {
  kbBuildKeywordGraph.checked = true;
  return window.startKbConstruction();
};

window.activateLatestStagedKb = async function activateLatestStagedKb() {
  show("kbStatus", "Replacing the active KB with the latest successful staged KB.");
  try {
    const data = await jsonRequest("/benchmark-workbench/kb/activate-latest", {});
    kbOutput.textContent = pretty(data);
    show("kbStatus", data.message || "Latest staged KB activated.");
    await window.loadKbStatus();
  } catch (error) {
    kbOutput.textContent = String(error);
    show("kbStatus", "Staged KB activation failed.");
  }
};

window.mergeLatestStagedKb = async function mergeLatestStagedKb() {
  show("kbStatus", "Merging the latest successful staged KB into the active KB and rebuilding the active index.");
  try {
    const data = await jsonRequest("/benchmark-workbench/kb/merge-latest", {});
    kbOutput.textContent = pretty(data);
    show("kbStatus", data.message || "Latest staged KB merged into the active KB.");
    await window.loadKbStatus();
  } catch (error) {
    kbOutput.textContent = String(error);
    show("kbStatus", "Staged KB merge failed.");
  }
};

window.openActiveKeywordGraph = function openActiveKeywordGraph() {
  window.open("/benchmark-workbench/kb/active-keyword-graph", "_blank", "noopener");
};

window.openLatestStagedKeywordGraph = function openLatestStagedKeywordGraph() {
  window.open("/benchmark-workbench/kb/staged/latest-keyword-graph", "_blank", "noopener");
};

async function pollKbConstructionJob() {
  if (!currentKbJobId) return;
  try {
    const response = await fetch(`/benchmark-workbench/kb/jobs/${currentKbJobId}`);
    const data = await parseJsonResponse(response, {allowApplicationError: true});
    kbOutput.textContent = pretty(data);
    const logLine = latestKbLogLine(data);
    const detail = logLine ? ` Latest log: ${logLine}` : "";
    show("kbStatus", `${data.message || data.status || "Staged KB job running."}${detail}`);
    if (["success", "error"].includes(data.status)) {
      kbBuildBtn.disabled = false;
      currentKbJobId = null;
      if (kbPollTimer) clearTimeout(kbPollTimer);
      kbPollTimer = null;
      await window.loadKbStatus();
      return;
    }
    kbPollTimer = setTimeout(pollKbConstructionJob, 2000);
  } catch (error) {
    kbOutput.textContent = String(error);
    show("kbStatus", "Staged KB job polling failed.");
    kbBuildBtn.disabled = false;
    currentKbJobId = null;
  }
}

window.applyBenchmarkPreset = function applyBenchmarkPreset(preset) {
  const cleanChunks = "petroleum_kb/kb_core/data/processed/chunks.jsonl";
  const presets = {
    reviewed_single: {
      benchmarkName: "benchmark.reviewed.single.200.en.final",
      benchmarkMode: "reviewed",
      benchmarkStyle: "semantic",
      answerFormat: "open",
      questionCount: 200,
      ratio1: 1,
      ratio2: 0,
      quality: 0.7
    },
    reviewed_twodoc: {
      benchmarkName: "benchmark.reviewed.twodoc.200.en.final",
      benchmarkMode: "reviewed",
      benchmarkStyle: "semantic",
      answerFormat: "open",
      questionCount: 200,
      ratio1: 0,
      ratio2: 1,
      quality: 0.7
    },
    specific_open: {
      benchmarkName: "benchmark.reviewed.specific_fact.single.open.200.en",
      benchmarkMode: "reviewed",
      benchmarkStyle: "specific_fact",
      answerFormat: "open",
      questionCount: 200,
      ratio1: 1,
      ratio2: 0,
      quality: 0.8
    },
    specific_mcq: {
      benchmarkName: "benchmark.reviewed.specific_fact.single.mcq.100.en",
      benchmarkMode: "reviewed",
      benchmarkStyle: "specific_fact",
      answerFormat: "mcq",
      questionCount: 100,
      ratio1: 1,
      ratio2: 0,
      quality: 0.8
    },
    naive_single: {
      benchmarkName: "benchmark.naive_baseline.single.200.en.final",
      benchmarkMode: "naive_baseline",
      benchmarkStyle: "semantic",
      answerFormat: "open",
      questionCount: 200,
      ratio1: 1,
      ratio2: 0,
      quality: 0.7
    },
    naive_twodoc: {
      benchmarkName: "benchmark.naive_baseline.twodoc.200.en.final",
      benchmarkMode: "naive_baseline",
      benchmarkStyle: "semantic",
      answerFormat: "open",
      questionCount: 200,
      ratio1: 0,
      ratio2: 1,
      quality: 0.7
    }
  };
  const cfg = presets[preset];
  if (!cfg) return;
  benchmarkName.value = cfg.benchmarkName;
  benchmarkMode.value = cfg.benchmarkMode;
  benchmarkStyle.value = cfg.benchmarkStyle;
  answerFormat.value = cfg.answerFormat;
  questionCount.value = cfg.questionCount;
  ratio1.value = cfg.ratio1.toFixed(2);
  ratio2.value = cfg.ratio2.toFixed(2);
  quality.value = cfg.quality;
  language.value = "en";
  writeMode.value = "new";
  sourceMode.value = "existing_chunks";
  chunksPath.value = cleanChunks;
  window.syncSourceMode();
  show("generationStatus", `Preset loaded: ${cfg.benchmarkName}`);
};

window.generateBenchmark = async function generateBenchmark() {
  show("generationStatus", "Starting benchmark generation job.");
  try {
    const data = await jsonRequest("/benchmark-workbench/benchmark/start", {
      benchmark_name: benchmarkName.value,
      write_mode: writeMode.value,
      benchmark_mode: benchmarkMode.value,
      source_mode: sourceMode.value,
      chunks_path: chunksPath.value || null,
      pdf_dir: pdfDir.value || null,
      pdf_parser_backend: pdfParserBackend.value,
      chunk_mode: chunkMode.value,
      benchmark_style: benchmarkStyle.value,
      answer_format: answerFormat.value,
      question_count: Number(questionCount.value),
      language: language.value,
      quality_min_score: Number(quality.value),
      single_doc_ratio: Number(ratio1.value),
      two_doc_ratio: Number(ratio2.value)
    });
    currentBenchmarkJobId = data.job_id;
    generateBtn.disabled = true;
    stopBtn.disabled = false;
    benchmarkOutput.textContent = pretty(data);
    show("generationStatus", `Benchmark job started: ${currentBenchmarkJobId}`);
    pollBenchmarkJob();
  } catch (error) {
    benchmarkOutput.textContent = String(error);
    show("generationStatus", "Benchmark generation request failed.");
  }
};

async function pollBenchmarkJob() {
  if (!currentBenchmarkJobId) return;
  try {
    const response = await fetch(`/benchmark-workbench/benchmark/jobs/${currentBenchmarkJobId}`);
    const data = await parseJsonResponse(response, {allowApplicationError: true});
    benchmarkOutput.textContent = pretty(data);
    show("generationStatus", data.message || data.status || "Benchmark job running.");
    if (["success", "error", "stopped"].includes(data.status)) {
      generateBtn.disabled = false;
      stopBtn.disabled = true;
      currentBenchmarkJobId = null;
      if (benchmarkPollTimer) clearTimeout(benchmarkPollTimer);
      benchmarkPollTimer = null;
      await window.loadOptions();
      return;
    }
    benchmarkPollTimer = setTimeout(pollBenchmarkJob, 1500);
  } catch (error) {
    benchmarkOutput.textContent = String(error);
    show("generationStatus", "Benchmark job polling failed.");
    generateBtn.disabled = false;
    stopBtn.disabled = true;
    currentBenchmarkJobId = null;
  }
}

window.stopBenchmark = async function stopBenchmark() {
  if (!currentBenchmarkJobId) return;
  try {
    const data = await jsonRequest(`/benchmark-workbench/benchmark/jobs/${currentBenchmarkJobId}/stop`, {});
    benchmarkOutput.textContent = pretty(data);
    show("generationStatus", "Stop requested for benchmark generation.");
  } catch (error) {
    benchmarkOutput.textContent = String(error);
    show("generationStatus", "Stop request failed.");
  }
};

window.chooseChunksFile = function chooseChunksFile() {
  chunksFilePicker.click();
};

window.usePickedChunksFile = function usePickedChunksFile(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  chunksPath.value = file.path || file.webkitRelativePath || file.name;
  show("generationStatus", "Selected chunks file name. If the backend cannot access it, paste the absolute path manually.");
};

window.choosePdfDirectory = function choosePdfDirectory() {
  pdfDirPicker.click();
};

window.usePickedPdfDirectory = function usePickedPdfDirectory(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const relative = file.webkitRelativePath || "";
  const folder = relative.includes("/") ? relative.split("/")[0] : "";
  if (file.path) {
    pdfDir.value = file.path;
    show("generationStatus", "Selected an absolute PDF folder path.");
    return;
  }
  if (!pdfDir.value && folder) {
    pdfDir.value = folder;
  }
  show("generationStatus", "Browser folder picker only provided a folder name. Paste the full folder path of your PDF corpus before generation.");
};

window.syncSourceMode = function syncSourceMode() {
  const pdfMode = sourceMode.value === "pdf_directory";
  chunksPath.disabled = pdfMode;
  chunksChooseBtn.disabled = pdfMode;
  if (pdfMode) {
    chunksPath.value = "";
    show("generationStatus", "PDF directory mode selected. Existing chunks JSONL path will be ignored.");
  }
};

bindManual("manualBtn", "manualModal");
mountLlmConsole("llmConfigPanel", {promptKey: "rag.query_refinement.user"});
window.loadKbStatus();
window.loadOptions();
window.syncSourceMode();
