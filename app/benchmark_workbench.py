from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import subprocess
import sys
import threading
import time
from shutil import copy2, copytree, rmtree
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.schemas import BenchmarkGenerationRequest
from app.schemas import KBConstructionRequest


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "petroleum_kb" / "eval"
DATASET_ROOT = EVAL_ROOT / "datasets"
CHUNKS_PATH = ROOT / "petroleum_kb" / "kb_core" / "data" / "processed" / "chunks.jsonl"
SOURCE_CHUNKS_ROOT = EVAL_ROOT / "source_chunks"
KB_STAGING_ROOT = ROOT / "outputs" / "kb_construction"
_BENCHMARK_JOBS: dict[str, "BenchmarkGenerationJob"] = {}
_BENCHMARK_JOBS_LOCK = threading.Lock()
_KB_JOBS: dict[str, "KBConstructionJob"] = {}
_KB_JOBS_LOCK = threading.Lock()


@dataclass
class BenchmarkGenerationJob:
    job_id: str
    request: BenchmarkGenerationRequest
    status: str = "queued"
    stage: str = "queued"
    message: str = "Queued benchmark generation."
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    command: list[str] = field(default_factory=list)
    stdout_lines: list[str] = field(default_factory=list)
    stderr_lines: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    returncode: int | None = None
    stop_requested: bool = False
    process: subprocess.Popen[str] | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "command": self.command,
            "returncode": self.returncode,
            "stop_requested": self.stop_requested,
            "stdout_tail": "\n".join(self.stdout_lines[-120:]),
            "stderr_tail": "\n".join(self.stderr_lines[-80:]),
            "result": self.result,
            "error": self.error,
        }

    def update(self, *, status: str | None = None, stage: str | None = None, message: str | None = None) -> None:
        if status is not None:
            self.status = status
        if stage is not None:
            self.stage = stage
        if message is not None:
            self.message = message
        self.updated_at = time.time()


@dataclass
class KBConstructionJob:
    job_id: str
    request: KBConstructionRequest
    build_id: str
    status: str = "queued"
    stage: str = "queued"
    message: str = "Queued staged KB construction."
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    stdout_lines: list[str] = field(default_factory=list)
    stderr_lines: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "build_id": self.build_id,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "request": self.request.model_dump(),
            "stdout_tail": "\n".join(self.stdout_lines[-120:]),
            "stderr_tail": "\n".join(self.stderr_lines[-80:]),
            "result": self.result,
            "error": self.error,
        }

    def update(self, *, status: str | None = None, stage: str | None = None, message: str | None = None) -> None:
        if status is not None:
            self.status = status
        if stage is not None:
            self.stage = stage
        if message is not None:
            self.message = message
        self.updated_at = time.time()


class _JobStreamMirror:
    def __init__(self, lines: list[str], original_stream: Any | None = None) -> None:
        self.lines = lines
        self.original_stream = original_stream
        self._buffer = ""

    def write(self, data: str) -> int:
        if not data:
            return 0
        if self.original_stream is not None:
            try:
                self.original_stream.write(data)
            except Exception:
                pass
        normalized = data.replace("\r", "\n")
        self._buffer += normalized
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.lines.append(line)
                del self.lines[:-400]
        return len(data)

    def flush(self) -> None:
        if self.original_stream is not None:
            try:
                self.original_stream.flush()
            except Exception:
                pass
        if self._buffer.strip():
            self.lines.append(self._buffer.strip())
            del self.lines[:-400]
            self._buffer = ""


def list_benchmark_datasets() -> list[dict[str, Any]]:
    if not DATASET_ROOT.exists():
        return []
    return [
        {
            "name": path.name,
            "path": str(path.relative_to(ROOT)),
            "size": path.stat().st_size,
        }
        for path in sorted(DATASET_ROOT.glob("*.jsonl"))
    ]


def kb_construction_status() -> dict[str, Any]:
    active_processed = ROOT / "petroleum_kb" / "kb_core" / "data" / "processed" / "chunks.jsonl"
    active_index_dir = ROOT / "petroleum_kb" / "kb_core" / "data" / "index"
    active_meta = active_index_dir / "petroleum_knowledge_meta.json"
    active_keyword_graph = (
        ROOT
        / "petroleum_kb"
        / "kb_core"
        / "data"
        / "analysis"
        / "keyword_graph_text_clean"
        / "topic_keyword_graph"
        / "keyword_graph.html"
    )
    active_meta_payload: dict[str, Any] = {}
    if active_meta.exists():
        try:
            active_meta_payload = json.loads(active_meta.read_text(encoding="utf-8"))
        except Exception:
            active_meta_payload = {"status": "unreadable", "path": str(active_meta)}
    builds = []
    if KB_STAGING_ROOT.exists():
        for path in sorted(KB_STAGING_ROOT.iterdir(), reverse=True):
            if not path.is_dir():
                continue
            summary_path = path / "summary.json"
            payload: dict[str, Any] = {"build_id": path.name, "path": str(path)}
            if summary_path.exists():
                try:
                    payload.update(json.loads(summary_path.read_text(encoding="utf-8")))
                except Exception:
                    payload["summary_status"] = "unreadable"
            builds.append(payload)
    latest_staged_build = builds[0] if builds else None
    latest_successful_staged_build = next((item for item in builds if item.get("status") == "success"), None)
    return {
        "status": "success",
        "active_kb": {
            "processed_chunks_file": str(active_processed),
            "processed_chunks_exists": active_processed.exists(),
            "processed_chunk_count": _count_jsonl(active_processed),
            "index_dir": str(active_index_dir),
            "index_meta": active_meta_payload,
            "keyword_graph_file": str(active_keyword_graph),
            "keyword_graph_exists": active_keyword_graph.exists(),
        },
        "staged_build_root": str(KB_STAGING_ROOT),
        "latest_staged_build": latest_staged_build,
        "latest_successful_staged_build": latest_successful_staged_build,
        "staged_builds": builds[:20],
        "safety": "Staged KB construction writes under outputs/kb_construction and does not activate or overwrite the current KB. After activation, the KB can be used directly for downstream RAG and QA benchmark construction.",
    }


def start_kb_construction_job(request: KBConstructionRequest) -> dict[str, Any]:
    build_id = f"{_safe_benchmark_name(request.build_name)}_{time.strftime('%Y%m%d_%H%M%S')}"
    job = KBConstructionJob(job_id=uuid4().hex, request=request, build_id=build_id)
    with _KB_JOBS_LOCK:
        _KB_JOBS[job.job_id] = job
    thread = threading.Thread(target=_run_kb_construction_job, args=(job,), daemon=True)
    thread.start()
    return job.snapshot()


def get_kb_construction_job(job_id: str) -> dict[str, Any]:
    with _KB_JOBS_LOCK:
        job = _KB_JOBS.get(job_id)
    if job is None:
        raise KeyError(f"Unknown KB construction job: {job_id}")
    return job.snapshot()


def latest_successful_staged_keyword_graph() -> Path:
    status = kb_construction_status()
    build = status.get("latest_successful_staged_build") or {}
    graph_path = (
        build.get("keyword_graph", {}) or {}
    ).get("topic_graph_html") or build.get("keyword_graph_file")
    if not graph_path:
        raise FileNotFoundError("No successful staged keyword graph is available.")
    path = Path(graph_path)
    if not path.exists():
        raise FileNotFoundError(f"Latest staged keyword graph not found: {path}")
    return path


def latest_successful_staged_build() -> dict[str, Any]:
    status = kb_construction_status()
    build = status.get("latest_successful_staged_build")
    if not build:
        raise FileNotFoundError("No successful staged KB build is available.")
    return build


def activate_latest_staged_kb() -> dict[str, Any]:
    build = latest_successful_staged_build()
    build_root = Path(build.get("build_root") or build.get("path") or "")
    staged_data_root = build_root / "petroleum_kb" / "data"
    if not staged_data_root.exists():
        raise FileNotFoundError(f"Staged KB data directory not found: {staged_data_root}")

    active_data_root = ROOT / "petroleum_kb" / "kb_core" / "data"
    active_data_root.mkdir(parents=True, exist_ok=True)

    copied_items: list[str] = []
    for name in ("raw", "processed", "metadata", "index", "analysis", "registry"):
        source = staged_data_root / name
        if not source.exists():
            continue
        target = active_data_root / name
        if target.exists():
            if target.is_dir():
                rmtree(target)
            else:
                target.unlink()
        if source.is_dir():
            copytree(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            copy2(source, target)
        copied_items.append(name)

    return {
        "status": "success",
        "activated_build_id": build.get("build_id"),
        "activated_from": str(build_root),
        "active_data_root": str(active_data_root),
        "copied_items": copied_items,
        "message": "Latest successful staged KB has replaced the active KB.",
    }


def merge_latest_staged_kb_into_active() -> dict[str, Any]:
    from petroleum_kb.config import KBConfig
    from petroleum_kb.kb_core.pipeline.build_index import build_vector_index

    build = latest_successful_staged_build()
    build_root = Path(build.get("build_root") or build.get("path") or "")
    staged_data_root = build_root / "petroleum_kb" / "data"
    if not staged_data_root.exists():
        raise FileNotFoundError(f"Staged KB data directory not found: {staged_data_root}")

    active_data_root = ROOT / "petroleum_kb" / "kb_core" / "data"
    active_data_root.mkdir(parents=True, exist_ok=True)

    active_processed = active_data_root / "processed"
    active_metadata = active_data_root / "metadata"
    active_registry = active_data_root / "registry"
    active_raw = active_data_root / "raw"
    active_analysis = active_data_root / "analysis"
    active_index = active_data_root / "index"

    staged_processed = staged_data_root / "processed"
    staged_metadata = staged_data_root / "metadata"
    staged_registry = staged_data_root / "registry"
    staged_raw = staged_data_root / "raw"

    active_chunks_path = active_processed / "chunks.jsonl"
    staged_chunks_path = staged_processed / "chunks.jsonl"
    if not staged_chunks_path.exists():
        raise FileNotFoundError(f"Staged chunks file not found: {staged_chunks_path}")

    def _load_jsonl_map(path: Path) -> dict[str, dict[str, Any]]:
        payload: dict[str, dict[str, Any]] = {}
        if not path.exists():
            return payload
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                chunk_id = str(item.get("chunk_id") or "").strip()
                if chunk_id:
                    payload[chunk_id] = item
        return payload

    def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for item in records:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _merge_registry_file(source: Path, target: Path) -> int:
        if not source.exists():
            return 0
        try:
            source_payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            target.parent.mkdir(parents=True, exist_ok=True)
            copy2(source, target)
            return 1
        target_payload: list[dict[str, Any]] = []
        if target.exists():
            try:
                loaded_target = json.loads(target.read_text(encoding="utf-8"))
                if isinstance(loaded_target, list):
                    target_payload = loaded_target
            except Exception:
                target_payload = []
        merged: dict[str, dict[str, Any]] = {}
        for item in target_payload:
            if isinstance(item, dict):
                key = str(item.get("path") or item.get("id") or json.dumps(item, sort_keys=True, ensure_ascii=False))
                merged[key] = item
        before = len(merged)
        if isinstance(source_payload, list):
            for item in source_payload:
                if isinstance(item, dict):
                    key = str(item.get("path") or item.get("id") or json.dumps(item, sort_keys=True, ensure_ascii=False))
                    merged[key] = item
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(list(merged.values()), ensure_ascii=False, indent=2), encoding="utf-8")
        return max(len(merged) - before, 0)

    def _copy_new_raw_files(source_root: Path, target_root: Path) -> tuple[int, int]:
        if not source_root.exists():
            return 0, 0
        copied = 0
        skipped = 0
        for src in source_root.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(source_root)
            dst = target_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                skipped += 1
                continue
            copy2(src, dst)
            copied += 1
        return copied, skipped

    active_chunks = _load_jsonl_map(active_chunks_path)
    staged_chunks = _load_jsonl_map(staged_chunks_path)
    active_count_before = len(active_chunks)
    staged_count = len(staged_chunks)
    for chunk_id, item in staged_chunks.items():
        active_chunks[chunk_id] = item
    merged_records = list(active_chunks.values())

    active_processed.mkdir(parents=True, exist_ok=True)
    active_metadata.mkdir(parents=True, exist_ok=True)
    _write_jsonl(active_chunks_path, merged_records)
    (active_metadata / "metadata.json").write_text(
        json.dumps(merged_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    raw_copied, raw_skipped = _copy_new_raw_files(staged_raw, active_raw)
    registry_added = _merge_registry_file(staged_registry / "sources.json", active_registry / "sources.json")

    config = KBConfig(project_root=ROOT)
    config.index_dir = active_index
    index_result = build_vector_index(active_chunks_path, config)
    if index_result.get("status") != "success":
        raise RuntimeError(f"Active KB index rebuild failed after merge: {index_result.get('error')}")

    graph_result = _build_staged_keyword_graph(
        input_path=active_chunks_path,
        outdir=active_analysis / "keyword_graph_text_clean",
    )

    return {
        "status": "success",
        "merged_build_id": build.get("build_id"),
        "merged_from": str(build_root),
        "active_data_root": str(active_data_root),
        "active_chunk_count_before": active_count_before,
        "staged_chunk_count": staged_count,
        "active_chunk_count_after": len(merged_records),
        "new_chunk_count": max(len(merged_records) - active_count_before, 0),
        "raw_files_copied": raw_copied,
        "raw_files_skipped": raw_skipped,
        "registry_entries_added": registry_added,
        "index": index_result,
        "keyword_graph": graph_result,
        "message": "Latest successful staged KB has been merged into the active KB and the active index has been rebuilt.",
    }




def _run_kb_construction_job(job: KBConstructionJob) -> None:
    from petroleum_kb.config import KBConfig
    from petroleum_kb.kb_core.pipeline.build_index import build_vector_index
    from petroleum_kb.kb_core.pipeline.ingest import ingest_directories

    request = job.request
    source_dir = Path(request.source_dir).expanduser()
    if not source_dir.exists() or not source_dir.is_dir():
        job.update(status="error", stage="validate_source", message="PDF source directory does not exist.")
        job.error = f"PDF source directory does not exist: {source_dir}"
        return
    build_root = KB_STAGING_ROOT / job.build_id
    config = KBConfig(project_root=build_root)
    config.pdf_parser_backend = request.pdf_parser_backend
    config.chunk_mode = request.chunk_mode
    config.strict_pdf_parser = request.pdf_parser_backend == "grobid"
    config.strict_scientific_sentence = request.chunk_mode == "scientific_sentence"
    config.chunk_token_size = request.chunk_token_size
    config.chunk_sentence_overlap = request.chunk_sentence_overlap
    if request.embedding_model:
        config.embedding_model_name = request.embedding_model
    config.embedding_local_files_only = bool(request.embedding_local_files_only)
    config.allow_hash_fallback = bool(request.allow_hash_fallback)
    job.update(status="running", stage="ingest", message="Parsing PDFs and generating staged KB chunks.")
    try:
        ingest_result = ingest_directories([source_dir], config, label_mode=request.label_mode, append=False)
        index_result: dict[str, Any] | None = None
        if request.build_index:
            job.update(status="running", stage="build_index", message="Building staged embedding index.")
            index_result = build_vector_index(config.processed_dir / "chunks.jsonl", config)
        keyword_graph_result: dict[str, Any] | None = None
        if request.build_keyword_graph:
            job.update(status="running", stage="keyword_graph", message="Building staged keyword graph.")
            keyword_graph_result = _build_staged_keyword_graph(
                input_path=config.processed_dir / "chunks.jsonl",
                outdir=config.data_root / "analysis" / "keyword_graph_text_clean",
            )
        summary = {
            "schema_version": "kb-construction-workbench.v1",
            "status": "success",
            "build_id": job.build_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source_dir": str(source_dir),
            "build_root": str(build_root),
            "processed_file": str(config.processed_dir / "chunks.jsonl"),
            "metadata_file": str(config.metadata_dir / "metadata.json"),
            "index_dir": str(config.index_dir),
            "keyword_graph_dir": str(config.data_root / "analysis" / "keyword_graph_text_clean"),
            "ingest": ingest_result,
            "index": index_result,
            "keyword_graph": keyword_graph_result,
            "activated": False,
            "safety": "This staged build was not copied to the active petroleum_kb data directory.",
        }
        build_root.mkdir(parents=True, exist_ok=True)
        (build_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        job.result = summary
        job.update(status="success", stage="complete", message="Staged KB construction completed without activating the KB.")
    except Exception as exc:
        job.error = str(exc)
        job.update(status="error", stage=job.stage, message="Staged KB construction failed.")


def _build_staged_keyword_graph(*, input_path: Path, outdir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "petroleum_kb" / "tools" / "keyword_graph_from_text.py"),
        "--input",
        str(input_path),
        "--outdir",
        str(outdir),
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    payload: dict[str, Any] = {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "topic_graph_html": str(outdir / "topic_keyword_graph" / "keyword_graph.html"),
        "core_graph_html": str(outdir / "core_domain_term_graph" / "keyword_graph.html"),
    }
    if completed.stdout.strip().startswith("{"):
        try:
            payload.update(json.loads(completed.stdout))
        except Exception:
            pass
    if completed.returncode != 0:
        raise RuntimeError(f"Keyword graph generation failed: {completed.stderr[-1000:]}")
    return payload


def _count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    count = 0
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
        return count
    except Exception:
        return None


def run_benchmark_generation(request: BenchmarkGenerationRequest) -> dict[str, Any]:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    name = _safe_benchmark_name(request.benchmark_name)
    output_path = DATASET_ROOT / f"{name}.jsonl"
    chunks_path, source_payload = _prepare_benchmark_source(request, benchmark_name=name)
    command = _build_generation_command(request, chunks_path=chunks_path, output_path=output_path)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=3600,
        check=False,
    )
    return _benchmark_result_payload(
        request=request,
        output_path=output_path,
        source_payload=source_payload,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def start_benchmark_generation_job(request: BenchmarkGenerationRequest) -> dict[str, Any]:
    job_id = uuid4().hex
    job = BenchmarkGenerationJob(job_id=job_id, request=request)
    with _BENCHMARK_JOBS_LOCK:
        _BENCHMARK_JOBS[job_id] = job
    thread = threading.Thread(target=_run_benchmark_generation_job, args=(job,), daemon=True)
    thread.start()
    return job.snapshot()


def get_benchmark_generation_job(job_id: str) -> dict[str, Any]:
    job = _get_benchmark_job(job_id)
    return job.snapshot()


def stop_benchmark_generation_job(job_id: str) -> dict[str, Any]:
    job = _get_benchmark_job(job_id)
    job.stop_requested = True
    job.update(message="Stop requested.")
    process = job.process
    if process is not None and process.poll() is None:
        process.terminate()
        job.update(status="stopping", stage=job.stage, message="Terminating benchmark generation process.")
    return job.snapshot()


def _get_benchmark_job(job_id: str) -> BenchmarkGenerationJob:
    with _BENCHMARK_JOBS_LOCK:
        job = _BENCHMARK_JOBS.get(job_id)
    if job is None:
        raise KeyError(f"Unknown benchmark generation job: {job_id}")
    return job


def _run_benchmark_generation_job(job: BenchmarkGenerationJob) -> None:
    try:
        DATASET_ROOT.mkdir(parents=True, exist_ok=True)
        request = job.request
        name = _safe_benchmark_name(request.benchmark_name)
        output_path = DATASET_ROOT / f"{name}.jsonl"
        job.update(status="running", stage="prepare_source", message="Preparing benchmark source.")
        chunks_path, source_payload = _prepare_benchmark_source(request, benchmark_name=name)
        if job.stop_requested:
            job.update(status="stopped", stage="stopped", message="Stopped before QA generation.")
            return

        command = _build_generation_command(request, chunks_path=chunks_path, output_path=output_path)
        job.command = command
        job.update(status="running", stage="generate_qa", message="Generating and reviewing QA pairs.")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        job.process = process

        def drain(stream: Any, target: list[str], is_stdout: bool) -> None:
            for line in iter(stream.readline, ""):
                cleaned = line.rstrip()
                if cleaned:
                    target.append(cleaned)
                    if is_stdout and ("progress:" in cleaned or cleaned.startswith("Wrote dataset")):
                        job.update(stage="generate_qa", message=cleaned)
                    else:
                        job.updated_at = time.time()
                if job.stop_requested and process.poll() is None:
                    process.terminate()
                    break

        stdout_thread = threading.Thread(target=drain, args=(process.stdout, job.stdout_lines, True), daemon=True)
        stderr_thread = threading.Thread(target=drain, args=(process.stderr, job.stderr_lines, False), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        returncode = process.wait()
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)
        job.returncode = returncode

        stdout = "\n".join(job.stdout_lines)
        stderr = "\n".join(job.stderr_lines)
        job.result = _benchmark_result_payload(
            request=request,
            output_path=output_path,
            source_payload=source_payload,
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )
        if job.stop_requested:
            job.update(status="stopped", stage="stopped", message="Benchmark generation stopped by user.")
        elif returncode == 0:
            job.update(status="success", stage="done", message="Benchmark generation completed.")
        else:
            job.update(status="error", stage="failed", message="Benchmark generation failed.")
    except Exception as exc:
        job.error = str(exc)
        job.result = {"status": "error", "error_type": type(exc).__name__, "message": str(exc)}
        job.update(status="error", stage="failed", message=str(exc))


def _build_generation_command(
    request: BenchmarkGenerationRequest,
    *,
    chunks_path: Path,
    output_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-B",
        str(EVAL_ROOT / "build_dataset.py"),
        "--chunks",
        str(chunks_path),
        "--out",
        str(output_path),
        "--num",
        str(request.question_count),
        "--mode",
        "mixed",
        "--benchmark-style",
        request.benchmark_style,
        "--answer-format",
        request.answer_format,
        "--language",
        request.language,
        "--quality-min-score",
        str(request.quality_min_score),
        "--seed",
        str(request.seed),
        "--progress-every",
        "1",
        "--doc-count-ratios",
        json.dumps(
            {
                "1": request.single_doc_ratio,
                "2": request.two_doc_ratio,
            }
        ),
        "--benchmark-mode",
        request.benchmark_mode,
    ]
    if request.write_mode == "append":
        command.append("--append")
    return command


def _benchmark_result_payload(
    *,
    request: BenchmarkGenerationRequest,
    output_path: Path,
    source_payload: dict[str, Any],
    command: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    return {
        "status": "success" if returncode == 0 else "error",
        "benchmark": {
            "name": output_path.name,
            "path": str(output_path.relative_to(ROOT)),
            "exists": output_path.exists(),
            "size": output_path.stat().st_size if output_path.exists() else 0,
        },
        "source": source_payload,
        "generation_policy": {
            "quality_review_required": True,
            "benchmark_mode": request.benchmark_mode,
            "quality_min_score": request.quality_min_score,
            "quality_filter_enabled": request.benchmark_mode == "reviewed",
            "pass_language_duplicate_gates_enabled": request.benchmark_mode == "reviewed",
            "rewrite_before_write": request.benchmark_mode != "naive_baseline",
            "benchmark_style": request.benchmark_style,
            "answer_format": request.answer_format,
            "review_dimensions": [
                "self_contained",
                "answerable",
                "objective",
                "specificity",
                "ambiguity",
                "evidence_locality",
                "reasoning_type",
                "difficulty",
                "hallucination_risk",
            ],
            "write_mode": request.write_mode,
            "doc_count_ratios": {
                "single": request.single_doc_ratio,
                "two_doc": request.two_doc_ratio,
            },
        },
        "command": command,
        "returncode": returncode,
        "stdout": stdout[-8000:],
        "stderr": stderr[-8000:],
    }


def _prepare_benchmark_source(request: BenchmarkGenerationRequest, *, benchmark_name: str) -> tuple[Path, dict[str, Any]]:
    if request.source_mode == "pdf_directory":
        if not request.pdf_dir:
            raise ValueError("pdf_dir is required when source_mode=pdf_directory")
        pdf_dir = _resolve_user_path(request.pdf_dir)
        if not pdf_dir.exists() or not pdf_dir.is_dir():
            raise ValueError(f"PDF directory does not exist: {pdf_dir}")
        chunks_path, ingest_result = _ingest_pdf_directory_for_benchmark(
            pdf_dir=pdf_dir,
            request=request,
            benchmark_name=benchmark_name,
        )
        return chunks_path, {
            "source_mode": request.source_mode,
            "pdf_dir": str(pdf_dir),
            "chunks_path": str(chunks_path.relative_to(ROOT)),
            "pdf_parser_backend": request.pdf_parser_backend,
            "chunk_mode": request.chunk_mode,
            "label_mode": "none",
            "ingest_result": ingest_result,
        }

    chunks_path = _resolve_user_path(request.chunks_path) if request.chunks_path else CHUNKS_PATH
    if not chunks_path.exists():
        raise ValueError(f"chunks JSONL does not exist: {chunks_path}")
    return chunks_path, {
        "source_mode": request.source_mode,
        "chunks_path": str(chunks_path),
        "note": "Benchmark generation starts from the existing active KB chunks.jsonl file.",
    }


def _resolve_user_path(value: str | None) -> Path:
    raw = str(value or "").strip().strip('"')
    if not raw:
        return CHUNKS_PATH
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [
        ROOT / path,
        Path.cwd() / path,
        ROOT / "petroleum_kb" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (ROOT / path).resolve()


def _ingest_pdf_directory_for_benchmark(
    *,
    pdf_dir: Path,
    request: BenchmarkGenerationRequest,
    benchmark_name: str,
) -> tuple[Path, dict[str, Any]]:
    kb_root = ROOT / "petroleum_kb"
    if str(kb_root) not in sys.path:
        sys.path.insert(0, str(kb_root))

    from config import KBConfig  # type: ignore
    from petroleum_kb.kb_core.pipeline.ingest import ingest_directories  # type: ignore

    config = KBConfig()
    output_root = SOURCE_CHUNKS_ROOT / _safe_benchmark_name(benchmark_name)
    config.data_root = output_root
    config.raw_dir = output_root / "raw"
    config.processed_dir = output_root / "processed"
    config.metadata_dir = output_root / "metadata"
    config.index_dir = output_root / "index"
    config.registry_dir = output_root / "registry"
    config.source_registry_file = config.registry_dir / "sources.json"
    config.bm25_index_file = config.index_dir / f"{config.index_name}.bm25.pkl"
    config.pdf_parser_backend = str(request.pdf_parser_backend or "grobid")
    config.chunk_mode = str(request.chunk_mode or "scientific_sentence")
    config.strict_pdf_parser = config.pdf_parser_backend == "grobid"
    config.strict_scientific_sentence = config.chunk_mode == "scientific_sentence"

    try:
        result = ingest_directories([pdf_dir], config, label_mode="none", append=False)
    except Exception as exc:
        raise RuntimeError(
            "PDF ingest failed before benchmark generation. "
            "Check that the PDF directory is reachable and parser dependencies are available. "
            f"parser={config.pdf_parser_backend}, chunk_mode={config.chunk_mode}, error={exc}"
        ) from exc

    chunks_path = config.processed_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise RuntimeError(f"PDF ingest did not produce chunks.jsonl: {chunks_path}")
    if int(result.get("chunks") or 0) <= 0:
        raise RuntimeError(
            "PDF ingest produced zero chunks. Check PDF readability, parser backend, and chunking settings."
        )
    return chunks_path, result


def _safe_benchmark_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip()).strip("._-")
    return cleaned or "benchmark.ui"
