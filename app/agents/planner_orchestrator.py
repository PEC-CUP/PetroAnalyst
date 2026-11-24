from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.agents.runtime_agent import PetroAnalystAgent
from app.agents.schemas import AnalysisConclusion, AnalysisRequest, CapabilitySchema, EvidenceBundle
from app.llm.prompt_registry import render_prompt
from app.llm.openai_expert_client import OpenAIExpertClient
from app.rag.kb_backend import (
    build_structured_rag_evidence,
    naive_rag_answer,
    iterative_review_rag_answer,
    qa_oriented_rag_answer,
    retrieve,
)
from app.tools.experimental_db.evidence_retriever import (
    attach_petroleum_expert_reasoning,
    build_sample_centered_database_evidence,
    build_tool_result_sample_database_evidence,
)
from app.tools.experimental_db import SampleDatabase
from app.tools.experimental_db.petroleum_fraction_spectral_evidence import (
    build_petroleum_fraction_spectral_workbook_evidence,
)
from app.utils.logger import setup_logger


logger = setup_logger(__name__)

PlannerTemplate = Literal["concept_qa", "knowledge_qa", "data_analysis", "processing_evaluation", "hybrid_analysis"]

_CHAT_MARKERS = ("hello", "hi", "help")
_GENERAL_CHAT_PHRASES = ("who are you", "introduce yourself", "what can you do", "capabilities")


def is_general_chat_query(query: str) -> bool:
    lowered = query.strip().lower()
    return lowered in _CHAT_MARKERS or any(marker in lowered for marker in _GENERAL_CHAT_PHRASES)


@dataclass(slots=True)
class PlanStep:
    step_id: str
    action: str
    status: str = "pending"
    detail: str = ""
    output: dict[str, Any] | None = None


@dataclass(slots=True)
class ExecutionPlan:
    template: PlannerTemplate
    goal: str
    status: str = "pending"
    steps: list[PlanStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "goal": self.goal,
            "status": self.status,
            "steps": [asdict(step) for step in self.steps],
            "metadata": self.metadata,
        }


class PlannerOrchestrator:
    _ROUTER_RAG_KEYWORDS = (
        "literature",
        "paper",
        "papers",
        "article",
        "articles",
        "reference",
        "references",
        "knowledge base",
        "kb",
        "standard",
        "standards",
        "definition",
        "concept",
        "principle",
        "method",
        "methods",
    )

    """Independent planner interface layered on top of the current PetroAnalyst release."""

    _WEAK_KG_ENTITY_TYPES = {"term", "method", "technique"}
    _EVALUATION_ONLY_FIELD_KEYS = {
        "validation_actual",
        "validation_actuals",
        "actual_for_validation",
        "actual_for_evaluation",
        "actual_for_evaluation_only",
        "actual",
        "actual_value",
        "measured",
        "measured_value",
        "absolute_error",
        "absolute_percentage_error",
        "absolute_percent_error",
        "relative_error",
        "relative_percentage_error",
        "relative_percent_error",
        "error",
        "error_pct",
        "ape",
        "mape",
        "actual_in_neighbor_range",
    }

    def __init__(self, agent: PetroAnalystAgent) -> None:
        self.agent = agent

    @classmethod
    def _strip_evaluation_only_fields(cls, value: Any) -> Any:
        """Remove validation-only fields before evidence is shown to any reasoning model."""

        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}
            for key, item in value.items():
                if str(key).lower() in cls._EVALUATION_ONLY_FIELD_KEYS:
                    continue
                cleaned[key] = cls._strip_evaluation_only_fields(item)
            return cleaned
        if isinstance(value, list):
            return [cls._strip_evaluation_only_fields(item) for item in value]
        return value

    def _build_analysis_request(
        self,
        *,
        query: str,
        session_id: str,
        template: PlannerTemplate,
        rag_mode_override: str | None = None,
        task_type_override: str | None = None,
    ) -> AnalysisRequest:
        session = self.agent.sessions.get_or_create(session_id)
        uploaded_artifacts = [str(item.path) for item in session.artifacts]
        constraints = {
            "has_uploaded_artifacts": bool(uploaded_artifacts),
            "last_tool_name": session.last_tool_name,
            "pending_tool_name": session.pending_tool_name,
        }
        return AnalysisRequest(
            query=query,
            session_id=session_id,
            template=template,
            rag_mode=rag_mode_override,
            task_type_override=task_type_override,
            uploaded_artifacts=uploaded_artifacts,
            constraints=constraints,
        )

    def _build_capability_schemas(
        self,
        *,
        template: PlannerTemplate,
        rag_mode_override: str | None = None,
    ) -> list[CapabilitySchema]:
        capabilities: list[CapabilitySchema] = []
        if template in {"concept_qa", "knowledge_qa", "processing_evaluation", "hybrid_analysis"}:
            capabilities.append(
                CapabilitySchema(
                    capability_id="knowledge.retrieve",
                    capability_type="knowledge",
                    description="Retrieve literature and KB evidence relevant to the current question.",
                    input_schema={"query": "string", "top_k": "integer"},
                    output_schema={"documents": "list"},
                )
            )
        if template in {"data_analysis", "hybrid_analysis"}:
            capabilities.extend(self._dynamic_tool_capabilities())
            capabilities.append(
                CapabilitySchema(
                    capability_id="database.experimental_history",
                    capability_type="database",
                    description=(
                        "Run uploaded petroleum fraction query workbooks through the Historical Experimental DB evidence path "
                        "without inserting the query sample into the historical database."
                    ),
                    input_schema={
                        "input_file": "Excel workbook",
                        "query": "property inference request",
                        "top_k": "integer",
                    },
                    output_schema={"neighbors": "list", "property_summary": "object", "expert_reasoning": "object"},
                    preconditions=["uploaded_query_workbook_available"],
                    artifacts_produced=["structured_experimental_db_evidence"],
                    invocation_mode="planner_routed_internal",
                )
            )
        return capabilities

    def _dynamic_tool_capabilities(self) -> list[CapabilitySchema]:
        """Planner-visible capabilities registered through Tool Onboarding.

        Built-in tools without the `tool.` namespace remain available through
        their existing hard-coded planner paths. Tool Onboarding providers use
        `tool.<name>` and can be selected dynamically.
        """

        capabilities: list[CapabilitySchema] = []
        registry = getattr(self.agent, "registry", None)
        if registry is None:
            return capabilities
        if hasattr(registry, "list_capabilities"):
            for schema in registry.list_capabilities():
                if not str(schema.capability_id).startswith("tool."):
                    continue
                capabilities.append(schema)
        elif hasattr(registry, "list_specs"):
            for spec in registry.list_specs():
                name = str(getattr(spec, "name", ""))
                if not name.startswith("tool."):
                    continue
                capabilities.append(
                    CapabilitySchema(
                        capability_id=name,
                        capability_type="tool",
                        description=str(getattr(spec, "description", "")),
                        input_schema=getattr(spec, "parameters", {}) if isinstance(getattr(spec, "parameters", {}), dict) else {},
                        output_schema={"status": "string", "tool_evidence": "object", "artifacts": "object"},
                        preconditions=list(getattr(spec, "required_upload_fields", ()) or ("uploaded_artifact_available",)),
                        artifacts_produced=[],
                        invocation_mode="dynamic_tool_registry",
                    )
                )
        return capabilities

    def _select_dynamic_tool(self, query: str) -> CapabilitySchema | None:
        ranked = self._rank_dynamic_tool_candidates(query)
        return ranked[0]["schema"] if ranked else None

    @staticmethod
    def _dynamic_tool_input_requirements(tool_schema: CapabilitySchema) -> dict[str, Any]:
        input_schema = tool_schema.input_schema if isinstance(tool_schema.input_schema, dict) else {}
        requirements = input_schema.get("x_petroanalyst_input_requirements")
        if isinstance(requirements, dict):
            return requirements
        return {
            "required_fields": input_schema.get("required", []),
            "properties": input_schema.get("properties", {}),
        }

    def _dynamic_tool_upload_answer(self, tool_schema: CapabilitySchema) -> str:
        requirements = self._dynamic_tool_input_requirements(tool_schema)
        lines = [
            f"`{tool_schema.capability_id}` requires an uploaded input file before execution.",
            "Required input:",
        ]
        file_type = requirements.get("file_type") if isinstance(requirements, dict) else None
        if file_type:
            lines.append(f"- File type: {file_type}.")
        required_sheets = requirements.get("required_sheets") if isinstance(requirements, dict) else None
        if isinstance(required_sheets, dict) and required_sheets:
            lines.append("- Required workbook sheets:")
            for sheet_name, columns in required_sheets.items():
                if isinstance(columns, list):
                    lines.append(f"  - `{sheet_name}`: {', '.join(str(col) for col in columns)}")
                else:
                    lines.append(f"  - `{sheet_name}`")
        notes = requirements.get("notes") if isinstance(requirements, dict) else None
        if isinstance(notes, list) and notes:
            lines.append("- Notes:")
            lines.extend(f"  - {note}" for note in notes)
        lines.append("Upload the workbook in the same chat session, then rerun the request.")
        return "\n".join(lines)

    @staticmethod
    def _dynamic_tool_template_input_file(tool_schema: CapabilitySchema) -> str | None:
        requirements = PlannerOrchestrator._dynamic_tool_input_requirements(tool_schema)
        templates = requirements.get("input_templates") if isinstance(requirements, dict) else None
        if not isinstance(templates, list):
            return None
        for template in templates:
            if not isinstance(template, dict):
                continue
            path = str(template.get("path") or "").strip()
            if path:
                return path
        return None

    def _dynamic_tool_upload_spec(self, tool_schema: CapabilitySchema) -> dict[str, Any]:
        spec = {
            "status": "awaiting_upload",
            "tool_name": tool_schema.capability_id,
            "accepted_file_types": [".xlsx", ".xls"],
            "instructions": self._dynamic_tool_upload_answer(tool_schema),
            "input_requirements": self._dynamic_tool_input_requirements(tool_schema),
        }
        input_schema = tool_schema.input_schema if isinstance(tool_schema.input_schema, dict) else {}
        input_text = json.dumps(input_schema, ensure_ascii=False).lower()
        if not ("excel" in input_text or "xlsx" in input_text or "xls" in input_text):
            spec["accepted_file_types"] = [".xlsx", ".xls", ".csv", ".json", ".txt"]
        template_input_file = self._dynamic_tool_template_input_file(tool_schema)
        if template_input_file:
            spec["template_input_file"] = template_input_file
        return spec

    def _rank_dynamic_tool_candidates(self, query: str) -> list[dict[str, Any]]:
        candidates = self._dynamic_tool_capabilities()
        if not candidates:
            return []
        query_tokens = self._tokenize_for_tool_selection(query)
        ranked: list[dict[str, Any]] = []
        for schema in candidates:
            text = (
                f"{schema.capability_id} {schema.description} "
                f"{json.dumps(schema.input_schema, ensure_ascii=False)} "
                f"{json.dumps(schema.output_schema, ensure_ascii=False)} "
                f"{' '.join(schema.preconditions or [])}"
            )
            tokens = self._tokenize_for_tool_selection(text)
            matched = sorted(query_tokens & tokens)
            score = len(matched)
            if score <= 0:
                continue
            ranked.append(
                {
                    "schema": schema,
                    "score": score,
                    "matched_terms": matched[:20],
                    "reason": "metadata_semantic_overlap",
                }
            )
        ranked.sort(key=lambda item: (-int(item["score"]), item["schema"].capability_id))
        return ranked

    @staticmethod
    def _tokenize_for_tool_selection(text: str) -> set[str]:
        normalized = re.sub(r"[_/\\:#.,(){}\[\]\-]+", " ", str(text).lower())
        tokens = set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", normalized))
        # Mixed-language queries can be split incorrectly around underscored tool names.
        # Keep ASCII token extraction enabled to preserve tool identifiers such as hydrocarbon_type.
        tokens.update(re.findall(r"[a-z0-9]+", normalized))
        stopwords = {"tool", "input", "output", "file", "string", "object", "data", "analysis", "calculate"}
        return {token for token in tokens if len(token) >= 2 and token not in stopwords}

    @staticmethod
    def _empty_evidence_bundle(capabilities: list[CapabilitySchema]) -> EvidenceBundle:
        return EvidenceBundle(capabilities_used=[item.to_dict() for item in capabilities])

    @staticmethod
    def _build_conclusion(
        *,
        direct_findings: dict[str, Any] | None = None,
        database_inference: dict[str, Any] | None = None,
        knowledge_interpretation: dict[str, Any] | None = None,
        uncertainty: dict[str, Any] | None = None,
        confidence: str | None = None,
    ) -> AnalysisConclusion:
        return AnalysisConclusion(
            direct_findings=direct_findings or {},
            database_inference=database_inference or {},
            knowledge_interpretation=knowledge_interpretation or {},
            uncertainty=uncertainty or {},
            confidence=confidence,
        )

    @staticmethod
    def _is_tool_enabled_mode(rag_mode_override: str | None) -> bool:
        return False

    @staticmethod
    def _uses_kg_mode(rag_mode_override: str | None) -> bool:
        return False

    @classmethod
    def _is_weak_kg_type(cls, value: Any) -> bool:
        return str(value or "").strip().lower() in cls._WEAK_KG_ENTITY_TYPES

    @classmethod
    def _is_semantic_kg_entity(cls, entity: Any) -> bool:
        return isinstance(entity, dict) and not cls._is_weak_kg_type(entity.get("type"))

    @classmethod
    def _is_semantic_kg_edge(cls, edge: Any) -> bool:
        if not isinstance(edge, dict):
            return False
        source_type = edge.get("source_type")
        target_type = edge.get("target_type")
        if not str(source_type or "").strip() and not str(target_type or "").strip():
            return False
        return not (cls._is_weak_kg_type(source_type) and cls._is_weak_kg_type(target_type))

    @classmethod
    def _semantic_kg_path_count(cls, paths: Any) -> int:
        if not isinstance(paths, list):
            return 0
        count = 0
        for path in paths:
            if not isinstance(path, dict):
                continue
            edges = path.get("edges")
            if isinstance(edges, list) and any(cls._is_semantic_kg_edge(edge) for edge in edges):
                count += 1
        return count

    def _latest_analysis_artifact(self, session_id: str) -> Path | None:
        preferred = self.agent.sessions.latest_artifact(
            session_id,
            file_types=(".xlsx", ".xls", ".csv", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pdf"),
            role="input",
        )
        if preferred and Path(preferred.path).exists():
            return Path(preferred.path).resolve()
        artifact = self.agent.sessions.latest_artifact(
            session_id,
            file_types=(".xlsx", ".xls", ".csv", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pdf"),
        )
        if artifact and Path(artifact.path).exists():
            return Path(artifact.path).resolve()
        return None

    def _latest_experimental_db_query_workbook(self, session_id: str) -> Path | None:
        artifact = self.agent.sessions.latest_artifact(session_id, file_types=(".xlsx", ".xls"), role="input")
        if artifact and Path(artifact.path).exists():
            return Path(artifact.path).resolve()
        artifact = self.agent.sessions.latest_artifact(session_id, file_types=(".xlsx", ".xls"))
        if artifact and Path(artifact.path).exists():
            return Path(artifact.path).resolve()
        return None

    @staticmethod
    def _is_experimental_db_property_query(query: str) -> bool:
        normalized = str(query or "").lower()
        db_markers = (
            "experimental db",
            "experimental database",
            "historical db",
            "historical database",
            "historical experimental",
            "training-free",
            "database evidence",
            "db evidence",




        )
        property_markers = (
            "property",
            "infer",
            "inference",
            "predict",
            "density",
            "distillation",
            "boiling",
            "bp_",
            "bp ",
            "saturates",
            "aromatics",
            "gc-fid",
            "gc fid",
            "ir",
            "spectrum",
            "spectral",










        )
        if any(marker in normalized for marker in db_markers):
            return True
        return any(marker in normalized for marker in ("gc-fid", "gc fid", "ir", "spectrum", "spectral")) and any(
            marker in normalized for marker in property_markers
        )

    @staticmethod
    def _parse_top_k(query: str, *, default: int = 10) -> int:
        match = re.search(r"(?:top[-_\s]*k|k)\s*[=:]?\s*(\d+)", str(query or ""), flags=re.I)
        if not match:
            return default
        return max(1, min(50, int(match.group(1))))

    def _experimental_db_upload_spec(self, *, query: str) -> dict[str, Any]:
        template_file = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "experimental_db"
            / "templates"
            / "petroanalyst_experimental_sample_template.xlsx"
        ).resolve()
        instructions = (
            "`database.experimental_history` requires an uploaded petroleum fraction query workbook before execution.\n"
            "Required input:\n"
            "- File type: .xlsx Excel workbook.\n"
            "- Recommended sheets: sample_information, gc_fid, and ir_spectrum.\n"
            "- The uploaded workbook is treated as a transient query sample and is not inserted into the historical database.\n"
            "Upload the workbook in the same chat session, then rerun the property-inference request."
        )
        spec: dict[str, Any] = {
            "status": "awaiting_upload",
            "tool_name": "database.experimental_history",
            "accepted_file_types": [".xlsx", ".xls"],
            "instructions": instructions,
            "input_requirements": {
                "file_type": ".xlsx Excel workbook",
                "required_workbook_sheets": ["sample_information", "gc_fid", "ir_spectrum"],
                "query_sample_policy": "transient_query_not_inserted_into_historical_db",
            },
        }
        if template_file.exists():
            spec["template_input_file"] = str(template_file)
        return spec

    @staticmethod
    def _default_sample_database_root() -> Path:
        return Path("data/experimental_db").resolve()

    def _sample_db_workspace(self) -> Path:
        return self._default_sample_database_root()

    def _extract_gc_summary(self, match_result: dict[str, Any]) -> dict[str, Any]:
        return {
            "matched_ratio": match_result.get("matched_ratio"),
            "matched_peaks": match_result.get("matched_peaks"),
            "total_peaks": match_result.get("total_peaks"),
            "unique_compounds": match_result.get("unique_compounds"),
            "output_excel": match_result.get("output_excel"),
        }

    def _maybe_build_database_evidence(
        self,
        sample_id: str | None,
        *,
        features: dict[str, Any] | None = None,
        tool_result: dict[str, Any] | None = None,
        query: str | None = None,
        top_k: int = 4,
    ) -> dict[str, Any]:
        sample_db_root = self._sample_db_workspace()
        if not sample_db_root.exists():
            return {
                "status": "skipped",
                "message": "sample_centered_database_workspace_missing",
                "sample_id": str(sample_id or "").strip(),
                "source": "sample_centered_experimental_db",
            }
        try:
            return build_sample_centered_database_evidence(
                sample_db_root,
                sample_id=str(sample_id or "").strip() or None,
                features=features,
                tool_result=tool_result,
                query=query,
                top_k=top_k,
            )
        except Exception as exc:
            logger.warning("Sample-centered database evidence failed: %s", exc)
            return {
                "status": "error",
                "message": str(exc),
                "sample_id": str(sample_id or "").strip(),
                "source": "sample_centered_experimental_db",
            }

    def _database_evidence_from_tool_result(
        self,
        *,
        tool_result: dict[str, Any],
        capability_id: str,
        query: str | None = None,
        top_k: int = 4,
    ) -> dict[str, Any]:
        sample_db_root = self._sample_db_workspace()
        if not sample_db_root.exists():
            return {
                "status": "skipped",
                "message": "sample_centered_database_workspace_missing",
                "source": "sample_centered_experimental_db",
                "capability_id": capability_id,
            }
        try:
            return build_tool_result_sample_database_evidence(
                sample_db_root,
                tool_result=tool_result,
                capability_id=capability_id,
                query=query,
                top_k=top_k,
            )
        except Exception as exc:
            logger.warning("Sample-centered tool-result database evidence failed: %s", exc)
            return {
                "status": "error",
                "message": str(exc),
                "source": "sample_centered_experimental_db",
                "capability_id": capability_id,
            }

    @staticmethod
    def _detect_evidence_conflicts(evidence: EvidenceBundle) -> list[dict[str, Any]]:
        """Detect simple numeric conflicts across tool and DB evidence.

        This is intentionally conservative: it only flags same-named numeric
        properties when a direct tool/session value diverges materially from the
        historical DB estimate. Broader semantic conflict checks can be added
        later without changing the Evidence Bundle contract.
        """

        tool = evidence.tool_evidence if isinstance(evidence.tool_evidence, dict) else {}
        database = evidence.database_evidence if isinstance(evidence.database_evidence, dict) else {}
        if not tool or not database:
            return []
        tool_values: dict[str, float] = {}
        for container_key in ("properties", "bulk_properties", "structured_result", "result"):
            PlannerOrchestrator._collect_named_numeric_values(tool.get(container_key), tool_values)
        PlannerOrchestrator._collect_named_numeric_values(tool.get("structured_tool_evidence"), tool_values)
        db_summary = database.get("property_summary") if isinstance(database.get("property_summary"), dict) else {}
        conflicts: list[dict[str, Any]] = []
        for name, summary in db_summary.items():
            if not isinstance(summary, dict):
                continue
            db_value = PlannerOrchestrator._coerce_float(summary.get("weighted_estimate"))
            if db_value is None:
                continue
            safe_name = PlannerOrchestrator._safe_compare_key(name)
            for tool_name, tool_value in tool_values.items():
                if PlannerOrchestrator._safe_compare_key(tool_name) != safe_name:
                    continue
                scale = max(abs(db_value), abs(tool_value), 1.0)
                relative_delta = abs(db_value - tool_value) / scale
                if relative_delta >= 0.15:
                    conflicts.append(
                        {
                            "property": name,
                            "tool_value": tool_value,
                            "database_estimate": db_value,
                            "relative_delta": round(relative_delta, 4),
                            "severity": "warning",
                        }
                    )
        return conflicts

    @staticmethod
    def _collect_named_numeric_values(value: Any, output: dict[str, float]) -> None:
        if isinstance(value, dict):
            label = value.get("property_name") or value.get("name") or value.get("item") or value.get("component")
            numeric = (
                PlannerOrchestrator._coerce_float(value.get("value"))
                or PlannerOrchestrator._coerce_float(value.get("content"))
                or PlannerOrchestrator._coerce_float(value.get("half_up_sum"))
                or PlannerOrchestrator._coerce_float(value.get("weighted_estimate"))
            )
            if label and numeric is not None:
                output[str(label)] = numeric
            for item in value.values():
                PlannerOrchestrator._collect_named_numeric_values(item, output)
        elif isinstance(value, list):
            for item in value:
                PlannerOrchestrator._collect_named_numeric_values(item, output)

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            if value in (None, "") or isinstance(value, bool):
                return None
            return float(str(value).replace("%", "").strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_compare_key(value: Any) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())

    def _attach_conflict_warnings(self, evidence: EvidenceBundle) -> list[dict[str, Any]]:
        conflicts = self._detect_evidence_conflicts(evidence)
        if conflicts:
            evidence.warnings.append("evidence_conflict_detected")
            evidence.database_evidence = {
                **(evidence.database_evidence or {}),
                "evidence_conflicts": conflicts,
            }
            if "evidence_conflict_detected" not in evidence.uncertainty_flags:
                evidence.uncertainty_flags.append("evidence_conflict_detected")
        return conflicts

    def _safe_retrieve(self, query: str, top_k: int, *, retrieval_overrides: dict | None = None) -> list[dict]:
        try:
            return retrieve(query, top_k=top_k, retrieval_overrides=retrieval_overrides)
        except Exception as exc:
            logger.warning("Planner retrieve fallback triggered: %s", exc)
            return []

    def _safe_rag_answer(
        self,
        query: str,
        top_k: int,
        *,
        rag_mode_override: str | None = None,
        retrieval_overrides: dict | None = None,
    ) -> dict:
        try:
            rag_mode = (rag_mode_override or "auto").strip().lower()
            if rag_mode == "naive":
                return naive_rag_answer(
                    query,
                    top_k=top_k,
                    retrieval_overrides=retrieval_overrides,
                )
            if rag_mode == "iterative_review":
                return iterative_review_rag_answer(query, retrieval_overrides=retrieval_overrides)
            if rag_mode == "qa_oriented" or (
                rag_mode == "auto" and bool(getattr(self.agent.settings, "advanced_rag_enabled", False))
            ):
                return qa_oriented_rag_answer(query, retrieval_overrides=retrieval_overrides)
            return naive_rag_answer(query, top_k=top_k, retrieval_overrides=retrieval_overrides)
        except Exception as exc:
            logger.warning("Planner rag_answer fallback triggered: %s", exc)
            return {
                "used_model": self.agent.settings.deepseek_chat_model,
                "answer": "",
                "raw_result": [],
                "error": str(exc),
            }

    def answer_query(
        self,
        query: str,
        *,
        session_id: str | None = None,
        task_type_override: str | None = None,
        template_override: str | None = None,
        rag_mode_override: str | None = None,
        retrieval_overrides: dict | None = None,
    ) -> dict:
        session = self.agent.sessions.get_or_create(session_id)

        template = self._classify_template(
            query=query,
            session=session,
            template_override=template_override,
            rag_mode_override=rag_mode_override,
            task_type_override=task_type_override,
        )
        analysis_request = self._build_analysis_request(
            query=query,
            session_id=session.session_id,
            template=template,
            rag_mode_override=rag_mode_override,
            task_type_override=task_type_override,
        )
        plan = self._build_plan(template=template, query=query)
        try:
            result = self._execute_plan(
                plan=plan,
                analysis_request=analysis_request,
                query=query,
                session_id=session.session_id,
                task_type_override=task_type_override,
                rag_mode_override=rag_mode_override,
                retrieval_overrides=retrieval_overrides,
            )
            return result
        except Exception as exc:
            logger.exception("Planner execution failed.")
            answer = f"Planner execution failed: {exc}"
            self.agent.sessions.set_last_plan(session.session_id, plan.to_dict())
            self.agent._record_turn(session.session_id, query, answer)
            return {
                "session_id": session.session_id,
                "task_type": "planner",
                "used_model": "none",
                "used_tools": [],
                "answer": answer,
                "raw_result": {
                    "planner": plan.to_dict(),
                    "planner_error": str(exc),
                },
            }
    def _classify_template(
        self,
        *,
        query: str,
        session,
        template_override: str | None = None,
        rag_mode_override: str | None = None,
        task_type_override: str | None = None,
    ) -> PlannerTemplate:
        override = (template_override or "").strip().lower()
        if override and override != "auto":
            return override  # type: ignore[return-value]

        if self._existing_dynamic_tool_result(session) is not None and self._is_tool_result_followup_query(query):
            return "hybrid_analysis"

        if self._is_experimental_db_property_query(query):
            return "data_analysis"

        if self._rank_dynamic_tool_candidates(query):
            return "data_analysis"

        explicit_rag_mode = (rag_mode_override or "").strip().lower()
        explicit_rag_modes = {
            "naive",
            "qa_oriented",
            "iterative_review",
        }
        requested_task = (task_type_override or "").strip().lower()
        if (
            explicit_rag_mode in explicit_rag_modes
            and requested_task in {"", "auto", "rag"}
            and not session.artifacts
            and not session.last_tool_result
            and (explicit_rag_mode == "iterative_review" or not self._rank_dynamic_tool_candidates(query))
        ):
            return "knowledge_qa"

        try:
            raw = self.agent.llm.chat(self._build_template_prompt(query=query, session=session))
            match = re.search(r"\{.*\}", raw, flags=re.S)
            if match:
                data = json.loads(match.group(0))
                template = str(data.get("template", "")).strip().lower()
                if template in {"concept_qa", "knowledge_qa", "data_analysis", "processing_evaluation", "hybrid_analysis"}:
                    return template  # type: ignore[return-value]
        except Exception as exc:
            logger.warning("Planner template classification fallback triggered: %s", exc)

        lowered = query.lower()
        if is_general_chat_query(query) or lowered in {"\u4f60\u597d", "hello", "hi", "help"}:
            return "concept_qa"
        if any(token in query for token in ("\u52a0\u5de5\u5efa\u8bae", "\u52a0\u5de5\u8bc4\u4ef7", "\u5de5\u827a\u8def\u7ebf", "\u52a0\u5de5\u98ce\u9669", "\u52a0\u5de5\u65b9\u6848", "\u6c14\u76f8\u8272\u8c31\u6570\u636e\u5904\u7406")):
            return "processing_evaluation"
        if session.artifacts and any(token in query for token in ("\u7ed3\u5408\u6587\u732e", "\u7ed3\u5408\u77e5\u8bc6\u5e93", "\u539f\u6599\u7279\u5f81", "\u7ec4\u6210\u7279\u5f81", "\u89e3\u91ca\u539f\u56e0", "\u4e0a\u4f20\u7684GC", "\u5206\u6790\u4e0a\u4f20")):
            return "hybrid_analysis"
        if session.last_tool_result or session.artifacts:
            return "data_analysis"
        if self._detect_task_type(query) == "rag":
            return "knowledge_qa"
        if any(token in lowered for token in ("\u539f\u7406", "\u5b9a\u4e49", "\u533a\u522b", "\u662f\u4ec0\u4e48", "\u4ec0\u4e48\u610f\u601d", "\u9002\u7528")):
            return "concept_qa"
        return "knowledge_qa"

    @staticmethod
    def _existing_dynamic_tool_result(session) -> tuple[str, dict[str, Any]] | None:
        tool_name = str(getattr(session, "last_tool_name", "") or "")
        tool_result = getattr(session, "last_tool_result", None)
        if not isinstance(tool_result, dict):
            return None
        if str(tool_result.get("status") or "").lower() != "success":
            return None
        if tool_name == "database.experimental_history":
            evidence = tool_result.get("database_evidence") or tool_result.get("evidence")
            return (tool_name, tool_result) if isinstance(evidence, dict) else None
        if not tool_name.startswith("tool."):
            return None
        if not isinstance(tool_result.get("tool_evidence"), dict):
            return None
        return tool_name, tool_result

    @staticmethod
    def _is_tool_result_followup_query(query: str) -> bool:
        normalized = str(query or "").lower()
        markers = (
            "result",
            "results",
            "analyze",
            "analyse",
            "interpret",
            "explain",
            "summarize",
            "summary",
            "previous",
            "returned",
            "tool evidence",
            "\u8fd9\u4e2a\u7ed3\u679c",
            "\u8fd9\u4e9b\u7ed3\u679c",
            "\u4e0a\u9762",
            "\u4e0a\u8ff0",
            "\u521a\u624d",
            "\u524d\u9762",
            "\u7ed3\u679c",
            "\u5206\u6790",
            "\u89e3\u91ca",
            "\u89e3\u8bfb",
            "\u603b\u7ed3",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _format_knowledge_references_for_prompt(references: list[dict[str, Any]]) -> str:
        if not references:
            return "No knowledge references were retrieved."
        lines: list[str] = []
        for index, ref in enumerate(references[:8], start=1):
            label = str(ref.get("label") or f"DOC {index}")
            title = str(ref.get("title") or ref.get("source") or ref.get("doc_id") or "untitled reference")
            authors = ref.get("authors") or ref.get("author") or ""
            year = ref.get("year") or ""
            doi = ref.get("doi") or ""
            parts = [title]
            if authors:
                parts.append(f"authors={authors}")
            if year:
                parts.append(f"year={year}")
            if doi:
                parts.append(f"doi={doi}")
            lines.append(f"[{label}] " + "; ".join(str(part) for part in parts if str(part).strip()))
        return "\n".join(lines)

    @classmethod
    def _detect_task_type(cls, query: str) -> str:
        normalized = str(query or "").strip().lower()
        rag_keywords = (
            "literature",
            "paper",
            "papers",
            "article",
            "articles",
            "reference",
            "references",
            "knowledge base",
            "kb",
            "standard",
            "standards",
            "definition",
            "concept",
            "principle",
            "method",
            "methods",
        )
        if any(keyword in normalized for keyword in rag_keywords):
            return "rag"
        return "chat"

    def _build_template_prompt(self, *, query: str, session) -> str:
        has_artifact = "true" if bool(session.artifacts) else "false"
        return render_prompt(
            "planner.template_classifier",
            has_artifact=has_artifact,
            last_tool=repr(session.last_tool_name),
            query=query,
        )

    def _build_plan(self, *, template: PlannerTemplate, query: str) -> ExecutionPlan:
        if template == "concept_qa":
            steps = [
                PlanStep("s1", "classify_intent"),
                PlanStep("s2", "retrieve_knowledge"),
                PlanStep("s3", "compose_answer"),
            ]
        elif template == "knowledge_qa":
            steps = [
                PlanStep("s1", "classify_intent"),
                PlanStep("s2", "retrieve_knowledge"),
                PlanStep("s3", "reason_over_context"),
                PlanStep("s4", "compose_answer"),
            ]
        elif template == "data_analysis":
            steps = [
                PlanStep("s1", "classify_intent"),
                PlanStep("s2", "extract_parameters"),
                PlanStep("s3", "check_inputs"),
                PlanStep("s4", "propose_tool"),
                PlanStep("s5", "confirm_tool"),
                PlanStep("s6", "run_tool"),
                PlanStep("s7", "compose_answer"),
            ]
        elif template == "hybrid_analysis":
            steps = [
                PlanStep("s1", "classify_intent"),
                PlanStep("s2", "parse_input"),
                PlanStep("s3", "validate_input"),
                PlanStep("s4", "run_toolchain"),
                PlanStep("s5", "retrieve_knowledge"),
                PlanStep("s6", "compose_answer"),
            ]
        else:
            steps = [
                PlanStep("s1", "classify_intent"),
                PlanStep("s2", "extract_parameters"),
                PlanStep("s3", "retrieve_knowledge"),
                PlanStep("s4", "reason_over_context"),
                PlanStep("s5", "compose_answer"),
            ]
        return ExecutionPlan(template=template, goal=query, steps=steps)

    def _execute_plan(
        self,
        *,
        plan: ExecutionPlan,
        analysis_request: AnalysisRequest,
        query: str,
        session_id: str,
        task_type_override: str | None = None,
        rag_mode_override: str | None = None,
        retrieval_overrides: dict | None = None,
    ) -> dict:
        if plan.template == "concept_qa":
            return self._execute_concept_plan(
                plan=plan,
                analysis_request=analysis_request,
                query=query,
                session_id=session_id,
                retrieval_overrides=retrieval_overrides,
            )
        if plan.template == "knowledge_qa":
            return self._execute_knowledge_plan(
                plan=plan,
                analysis_request=analysis_request,
                query=query,
                session_id=session_id,
                rag_mode_override=rag_mode_override,
                retrieval_overrides=retrieval_overrides,
            )
        if plan.template == "data_analysis":
            return self._execute_data_analysis_plan(
                plan=plan,
                analysis_request=analysis_request,
                query=query,
                session_id=session_id,
                task_type_override=task_type_override,
                rag_mode_override=rag_mode_override,
                retrieval_overrides=retrieval_overrides,
            )
        if plan.template == "hybrid_analysis":
            return self._execute_hybrid_plan(
                plan=plan,
                analysis_request=analysis_request,
                query=query,
                session_id=session_id,
                rag_mode_override=rag_mode_override,
                retrieval_overrides=retrieval_overrides,
            )
        return self._execute_processing_plan(
            plan=plan,
            analysis_request=analysis_request,
            query=query,
            session_id=session_id,
            task_type_override=task_type_override,
            retrieval_overrides=retrieval_overrides,
        )

    def _execute_data_analysis_plan(
        self,
        *,
        plan: ExecutionPlan,
        analysis_request: AnalysisRequest,
        query: str,
        session_id: str,
        task_type_override: str | None = None,
        rag_mode_override: str | None = None,
        retrieval_overrides: dict | None = None,
    ) -> dict:
        capabilities = self._build_capability_schemas(template=plan.template, rag_mode_override=rag_mode_override)
        evidence = self._empty_evidence_bundle(capabilities)
        session = self.agent.sessions.get_or_create(session_id)
        plan.steps[0].status = "completed"
        plan.steps[1].status = "completed"
        plan.steps[1].output = {
            "semantic_route": "data_analysis",
            "session_has_artifact": bool(session.artifacts),
        }
        plan.steps[2].status = "completed"
        plan.steps[2].output = {
            "has_uploaded_files": bool(session.artifacts),
            "has_last_tool_result": bool(session.last_tool_result),
        }

        existing_dynamic_tool = self._existing_dynamic_tool_result(session)
        if existing_dynamic_tool is not None and self._is_tool_result_followup_query(query):
            return self._execute_hybrid_plan(
                plan=plan,
                analysis_request=analysis_request,
                query=query,
                session_id=session_id,
                rag_mode_override=rag_mode_override,
                retrieval_overrides=retrieval_overrides,
            )

        if self._is_experimental_db_property_query(query):
            return self._execute_experimental_db_workbook_plan(
                plan=plan,
                analysis_request=analysis_request,
                query=query,
                session_id=session_id,
                retrieval_overrides=retrieval_overrides,
            )

        dynamic_candidates = self._rank_dynamic_tool_candidates(query)
        if dynamic_candidates:
            if not session.artifacts:
                return self._execute_dynamic_tool_plan(
                    plan=plan,
                    analysis_request=analysis_request,
                    query=query,
                    session_id=session_id,
                    tool_schema=dynamic_candidates[0]["schema"],
                    retrieval_overrides=retrieval_overrides,
                )

            last_result: dict | None = None
            for candidate in dynamic_candidates[:3]:
                last_result = self._execute_dynamic_tool_plan(
                    plan=plan,
                    analysis_request=analysis_request,
                    query=query,
                    session_id=session_id,
                    tool_schema=candidate["schema"],
                    retrieval_overrides=retrieval_overrides,
                )
                tool_status = (
                    last_result.get("raw_result", {})
                    .get("evidence_bundle", {})
                    .get("tool_evidence", {})
                    .get("status")
                )
                if tool_status == "success":
                    return last_result
            if isinstance(last_result, dict):
                return last_result

        plan.steps[3].status = "completed"
        plan.steps[3].detail = "no_matching_dynamic_tool"
        plan.steps[4].status = "skipped"
        plan.steps[5].status = "skipped"
        plan.steps[6].status = "completed"
        plan.status = "failed"
        answer = "No reviewed analytical tool matched the current request in the current PetroAnalyst release."
        conclusion = self._build_conclusion(
            uncertainty={"planner_status": "no_matching_dynamic_tool"},
            confidence="low",
        )
        self.agent.sessions.set_last_plan(session_id, plan.to_dict())
        self.agent._record_turn(session_id, query, answer)
        return {
            "session_id": session_id,
            "task_type": "planner",
            "used_model": "none",
            "used_tools": [],
            "answer": answer,
            "raw_result": {
                "planner": plan.to_dict(),
                "analysis_request": analysis_request.to_dict(),
                "evidence_bundle": evidence.to_dict(),
                "analysis_conclusion": conclusion.to_dict(),
            },
        }

    def _execute_dynamic_tool_plan(
        self,
        *,
        plan: ExecutionPlan,
        analysis_request: AnalysisRequest,
        query: str,
        session_id: str,
        tool_schema: CapabilitySchema,
        retrieval_overrides: dict | None = None,
    ) -> dict:
        evidence = EvidenceBundle(capabilities_used=[tool_schema.to_dict()])
        sample_path = self._latest_analysis_artifact(session_id)
        session = self.agent.sessions.get_or_create(session_id)
        reusable_tool_result = (
            session.last_tool_result
            if session.last_tool_name == tool_schema.capability_id and isinstance(session.last_tool_result, dict)
            else None
        )
        upload_spec = (
            self._experimental_db_upload_spec(query=query)
            if tool_schema.capability_id == "database.experimental_history"
            else self._dynamic_tool_upload_spec(tool_schema)
        )
        if sample_path is None:
            tool_result = {
                "status": "awaiting_upload",
                "message": "missing_input_artifact",
                "input_requirements": upload_spec.get("input_requirements", {}),
            }
        elif reusable_tool_result and reusable_tool_result.get("status") == "success":
            tool_result = reusable_tool_result
        else:
            safe_tool_name = re.sub(r"[^a-z0-9._-]+", "_", tool_schema.capability_id.lower())
            out_dir = sample_path.parent / "orchestration_runs" / session_id / safe_tool_name
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "input_file": str(sample_path),
                "file_path": str(sample_path),
                "excel_file": str(sample_path),
                "output_dir": str(out_dir),
            }
            if tool_schema.capability_id == "database.experimental_history":
                payload["query"] = query
                payload["top_k"] = self._parse_top_k(query, default=10)
            tool_result = self.agent.registry.call(tool_schema.capability_id, payload)

        plan.steps[3].status = "completed"
        plan.steps[3].output = {
            "tool_name": tool_schema.capability_id,
            "selection_mode": "dynamic_capability_registry",
            "description": tool_schema.description,
            "execution_mode": "reused_last_tool_result" if tool_result is reusable_tool_result else "adapter_call",
        }
        plan.steps[4].status = "completed"
        plan.steps[4].detail = "dynamic_capability_execution"
        plan.steps[5].status = "completed" if tool_result.get("status") == "success" else "failed"

        tool_result_for_reasoning = self._strip_evaluation_only_fields(tool_result)
        tool_evidence = (
            tool_result_for_reasoning.get("tool_evidence")
            if isinstance(tool_result_for_reasoning.get("tool_evidence"), dict)
            else {}
        )
        artifacts = (
            tool_result_for_reasoning.get("artifacts")
            if isinstance(tool_result_for_reasoning.get("artifacts"), dict)
            else {}
        )
        structured_result = tool_evidence.get("structured_result") if isinstance(tool_evidence, dict) else None
        evidence.tool_evidence = {
            "capability_id": tool_schema.capability_id,
            "description": tool_schema.description,
            "status": tool_result_for_reasoning.get("status"),
            "message": tool_result_for_reasoning.get("message"),
            "structured_tool_evidence": tool_evidence,
            "structured_result": structured_result,
            "artifacts": artifacts,
            "result": tool_result_for_reasoning.get("result"),
            "experimental_record": tool_result_for_reasoning.get("experimental_record"),
        }
        dynamic_database_evidence = self._strip_evaluation_only_fields(
            self._database_evidence_from_tool_result(
                tool_result=tool_result,
                capability_id=tool_schema.capability_id,
                query=query,
            )
        )
        if tool_schema.capability_id == "database.experimental_history" and isinstance(tool_result.get("database_evidence"), dict):
            dynamic_database_evidence = self._strip_evaluation_only_fields(tool_result.get("database_evidence") or {})
        evidence.database_evidence = dynamic_database_evidence
        self._attach_conflict_warnings(evidence)
        if dynamic_database_evidence.get("status") not in {"success", "skipped"}:
            evidence.uncertainty_flags.append("database_evidence_unavailable")
        if tool_result.get("status") != "success":
            evidence.uncertainty_flags.append("dynamic_tool_failed")

        if sample_path is None:
            answer = str(upload_spec["instructions"])
            self.agent.sessions.set_pending_tool(session_id, tool_schema.capability_id)
        else:
            prompt = render_prompt(
                "planner.dynamic_tool_summary",
                query=query,
                capability_id=tool_schema.capability_id,
                description=tool_schema.description,
                tool_evidence=json.dumps(evidence.tool_evidence, ensure_ascii=False),
                database_evidence=json.dumps(dynamic_database_evidence, ensure_ascii=False),
            )
            try:
                answer = self.agent.llm.chat(prompt)
                used_model = self.agent.settings.deepseek_chat_model
            except Exception as exc:
                logger.warning("Dynamic tool summary LLM fallback triggered: %s", exc)
                answer = self._format_dynamic_tool_answer(
                    tool_schema=tool_schema,
                    tool_result=tool_result_for_reasoning,
                    evidence=evidence,
                )
                used_model = "none"
        plan.steps[6].status = "completed"
        plan.status = "completed" if tool_result.get("status") == "success" else "awaiting_upload"

        conclusion = self._build_conclusion(
            direct_findings={
                "capability_id": tool_schema.capability_id,
                "status": tool_result.get("status"),
                "structured_result": structured_result,
                "artifacts": artifacts,
            },
            database_inference=(
                dynamic_database_evidence.get("property_summary", {})
                if isinstance(dynamic_database_evidence, dict)
                else {}
            ),
            uncertainty={
                "dynamic_tool_status": tool_result.get("status"),
                "dynamic_tool_message": tool_result.get("message"),
                "database_evidence_status": dynamic_database_evidence.get("status") if isinstance(dynamic_database_evidence, dict) else None,
            },
            confidence="moderate" if tool_result.get("status") == "success" else "low",
        )
        self.agent.sessions.remember_tool_result(session_id, tool_schema.capability_id, tool_result)
        self.agent.sessions.set_last_plan(session_id, plan.to_dict())
        self.agent._record_turn(session_id, query, answer)
        return {
            "session_id": session_id,
            "task_type": "planner",
            "used_model": used_model if sample_path is not None else "none",
            "used_tools": [tool_schema.capability_id],
            "answer": answer,
            "raw_result": {
                **(upload_spec if sample_path is None else {}),
                "planner": plan.to_dict(),
                "analysis_request": analysis_request.to_dict(),
                "evidence_bundle": evidence.to_dict(),
                "analysis_conclusion": conclusion.to_dict(),
                "dynamic_tool_result": tool_result_for_reasoning,
            },
        }

    @staticmethod
    def _format_dynamic_tool_answer(
        *,
        tool_schema: CapabilitySchema,
        tool_result: dict[str, Any],
        evidence: EvidenceBundle,
    ) -> str:
        status = str(tool_result.get("status") or "unknown")
        message = str(tool_result.get("message") or "")
        lines = [
            f"Tool `{tool_schema.capability_id}` finished with status `{status}`.",
        ]
        if message:
            lines.append(f"Message: {message}.")

        tool_evidence = tool_result.get("tool_evidence") if isinstance(tool_result.get("tool_evidence"), dict) else {}
        structured_result = tool_evidence.get("structured_result") if isinstance(tool_evidence, dict) else None
        if isinstance(structured_result, list) and structured_result:
            columns = sorted({str(key) for row in structured_result[:20] if isinstance(row, dict) for key in row.keys()})
            if columns:
                lines.extend(["", "Structured result preview:"])
                preview_columns = columns[:6]
                lines.append("| " + " | ".join(preview_columns) + " |")
                lines.append("|" + "|".join("---" for _ in preview_columns) + "|")
                for row in structured_result[:8]:
                    if not isinstance(row, dict):
                        continue
                    values = [str(row.get(column, "")) for column in preview_columns]
                    lines.append("| " + " | ".join(values) + " |")
                if len(structured_result) > 8:
                    lines.append(f"\nShowing 8 of {len(structured_result)} records.")
        elif structured_result:
            lines.extend(["", f"Structured result: {structured_result}"])

        artifacts = tool_result.get("artifacts") if isinstance(tool_result.get("artifacts"), dict) else {}
        if artifacts:
            lines.extend(["", "Generated artifacts:"])
            for name, path in artifacts.items():
                lines.append(f"- `{name}`: {path}")

        if evidence.uncertainty_flags:
            lines.extend(["", "Limitations: " + ", ".join(evidence.uncertainty_flags) + "."])
        return "\n".join(lines)

    def _execute_hybrid_plan(
        self,
        *,
        plan: ExecutionPlan,
        analysis_request: AnalysisRequest,
        query: str,
        session_id: str,
        rag_mode_override: str | None = None,
        retrieval_overrides: dict | None = None,
    ) -> dict:
        session = self.agent.sessions.get_or_create(session_id)
        existing_dynamic_tool = self._existing_dynamic_tool_result(session)
        if existing_dynamic_tool is None:
            plan.steps[0].status = "completed"
            for index in range(1, len(plan.steps)):
                plan.steps[index].status = "skipped"
            plan.status = "failed"
            answer = "No completed tool result is available in the current session for follow-up interpretation."
            conclusion = self._build_conclusion(
                uncertainty={"planner_status": "missing_previous_tool_result"},
                confidence="low",
            )
            self.agent.sessions.set_last_plan(session_id, plan.to_dict())
            self.agent._record_turn(session_id, query, answer)
            return {
                "session_id": session_id,
                "task_type": "planner",
                "used_model": "none",
                "used_tools": [],
                "answer": answer,
                "raw_result": {
                    "planner": plan.to_dict(),
                    "analysis_request": analysis_request.to_dict(),
                    "analysis_conclusion": conclusion.to_dict(),
                },
            }

        tool_name, tool_result = existing_dynamic_tool
        capabilities = self._build_capability_schemas(template=plan.template, rag_mode_override=analysis_request.rag_mode)
        evidence = self._empty_evidence_bundle(capabilities)
        tool_result_for_reasoning = self._strip_evaluation_only_fields(tool_result)
        tool_evidence = (
            tool_result_for_reasoning.get("tool_evidence")
            if isinstance(tool_result_for_reasoning.get("tool_evidence"), dict)
            else {}
        )
        artifacts = (
            tool_result_for_reasoning.get("artifacts")
            if isinstance(tool_result_for_reasoning.get("artifacts"), dict)
            else {}
        )
        structured_result = tool_evidence.get("structured_result")
        evidence.tool_evidence = {
            "capability_id": tool_name,
            "status": tool_result_for_reasoning.get("status"),
            "message": tool_result_for_reasoning.get("message"),
            "structured_tool_evidence": tool_evidence,
            "structured_result": structured_result,
            "artifacts": artifacts,
            "result": tool_result_for_reasoning.get("result"),
            "reused_from_session": True,
        }
        if tool_name == "database.experimental_history":
            evidence.database_evidence = self._strip_evaluation_only_fields(
                tool_result.get("database_evidence")
                or tool_result.get("evidence")
                or {}
            )
        else:
            evidence.database_evidence = self._strip_evaluation_only_fields(
                self._database_evidence_from_tool_result(
                    tool_result=tool_result,
                    capability_id=tool_name,
                    query=query,
                )
            )
        self._attach_conflict_warnings(evidence)

        plan.steps[0].status = "completed"
        if len(plan.steps) > 1:
            plan.steps[1].status = "completed"
            plan.steps[1].output = {"source": "session.last_tool_result", "tool_name": tool_name}
        if len(plan.steps) > 2:
            plan.steps[2].status = "completed"
            plan.steps[2].output = {"has_tool_result": True}
        if len(plan.steps) > 3:
            plan.steps[3].status = "completed"
            plan.steps[3].detail = "reuse_previous_tool_result"

        knowledge_query = (
            f"{query}\n"
            f"Existing analytical tool evidence ({tool_name}): "
            f"{json.dumps(evidence.tool_evidence, ensure_ascii=False)}"
        )
        rag_result = self._safe_rag_answer(
            knowledge_query,
            top_k=self.agent.settings.rag_top_k,
            rag_mode_override="qa_oriented",
            retrieval_overrides=retrieval_overrides,
        )
        structured_rag = build_structured_rag_evidence(rag_result.get("raw_result"), mode="qa_oriented")
        knowledge_raw = rag_result.get("raw_result", {}) if isinstance(rag_result.get("raw_result"), dict) else {}
        raw_evidences = knowledge_raw.get("evidences", []) if isinstance(knowledge_raw.get("evidences"), list) else []
        evidence_count = len(raw_evidences)
        knowledge_references = structured_rag.get("knowledge_evidence", {}).get("references", [])
        if not isinstance(knowledge_references, list):
            knowledge_references = []
        evidence.knowledge_evidence = {
            **structured_rag.get("knowledge_evidence", {}),
            "answer": rag_result.get("answer", ""),
            "evidence_count": evidence_count,
            "references": knowledge_references,
        }
        if len(plan.steps) > 4:
            plan.steps[4].status = "completed"
            plan.steps[4].output = {
                "knowledge_mode": "qa_oriented",
                "tool_name": tool_name,
                "knowledge_answer_present": bool(rag_result.get("answer")),
                "evidence_count": evidence_count,
            }

        synthesis_prompt = render_prompt(
            "planner.hybrid_synthesis",
            query=query,
            tool_name=tool_name,
            tool_evidence=json.dumps(evidence.tool_evidence, ensure_ascii=False),
            database_evidence=json.dumps(evidence.database_evidence, ensure_ascii=False),
            knowledge_answer=rag_result.get("answer", ""),
            knowledge_references=self._format_knowledge_references_for_prompt(knowledge_references),
        )
        answer = self.agent.llm.chat(synthesis_prompt)
        if len(plan.steps) > 5:
            plan.steps[5].status = "completed"
        plan.status = "completed"

        conclusion = self._build_conclusion(
            direct_findings={
                "tool_name": tool_name,
                "structured_result": structured_result,
                "artifacts": artifacts,
                "reused_tool_result": True,
            },
            database_inference=(
                evidence.database_evidence.get("property_summary", {})
                if isinstance(evidence.database_evidence, dict)
                else {}
            ),
            knowledge_interpretation={
                "knowledge_mode": "qa_oriented",
                "evidence_count": evidence_count,
            },
            uncertainty={
                "source": "previous_tool_result",
                "tool_status": tool_result_for_reasoning.get("status"),
                "database_evidence_status": evidence.database_evidence.get("status") if isinstance(evidence.database_evidence, dict) else None,
                "database_evidence_message": evidence.database_evidence.get("message") if isinstance(evidence.database_evidence, dict) else None,
            },
            confidence="moderate" if rag_result.get("answer") else "low",
        )
        self.agent.sessions.set_last_plan(session_id, plan.to_dict())
        self.agent._record_turn(session_id, query, answer)
        return {
            "session_id": session_id,
            "task_type": "planner",
            "used_model": self.agent.settings.deepseek_chat_model,
            "used_tools": [tool_name],
            "answer": answer,
            "raw_result": {
                "planner": plan.to_dict(),
                "analysis_request": analysis_request.to_dict(),
                "evidence_bundle": evidence.to_dict(),
                "analysis_conclusion": conclusion.to_dict(),
                "dynamic_tool_result": tool_result_for_reasoning,
                "knowledge": rag_result,
                "reused_previous_tool_result": True,
            },
        }

    def _execute_experimental_db_workbook_plan(
        self,
        *,
        plan: ExecutionPlan,
        analysis_request: AnalysisRequest,
        query: str,
        session_id: str,
        retrieval_overrides: dict | None = None,
    ) -> dict:
        capabilities = self._build_capability_schemas(template=plan.template)
        evidence = self._empty_evidence_bundle(capabilities)
        sample_path = self._latest_experimental_db_query_workbook(session_id)
        plan.steps[3].status = "completed"
        plan.steps[3].output = {
            "tool_name": "database.experimental_history",
            "selection_mode": "planner_experimental_db_property_query",
        }

        if sample_path is None:
            upload_spec = self._experimental_db_upload_spec(query=query)
            for index in (4, 5):
                if index < len(plan.steps):
                    plan.steps[index].status = "skipped"
            if len(plan.steps) > 6:
                plan.steps[6].status = "completed"
            plan.status = "awaiting_upload"
            answer = str(upload_spec["instructions"])
            self.agent.sessions.set_pending_tool(session_id, "database.experimental_history")
            self.agent.sessions.set_last_plan(session_id, plan.to_dict())
            self.agent._record_turn(session_id, query, answer)
            return {
                "session_id": session_id,
                "task_type": "planner",
                "used_model": "none",
                "used_tools": ["database.experimental_history"],
                "answer": answer,
                "raw_result": {
                    **upload_spec,
                    "planner": plan.to_dict(),
                    "analysis_request": analysis_request.to_dict(),
                    "evidence_bundle": evidence.to_dict(),
                },
            }

        top_k = self._parse_top_k(query, default=10)
        workspace = Path(__file__).resolve().parents[2] / "data" / "experimental_db"
        db_evidence_raw = build_petroleum_fraction_spectral_workbook_evidence(
            workbook=sample_path,
            template_definition=SampleDatabase(workspace).get_template_definition(),
            query=query,
            top_k=top_k,
        )
        db_evidence = self._strip_evaluation_only_fields(db_evidence_raw)
        try:
            db_evidence = attach_petroleum_expert_reasoning(db_evidence, OpenAIExpertClient())
        except Exception:
            logger.exception("Failed to attach expert reasoning for Experimental DB evidence.")
        evidence.database_evidence = db_evidence
        evidence.tool_evidence = {
            "capability_id": "database.experimental_history",
            "status": db_evidence.get("status"),
            "input_file": str(sample_path),
            "top_k": top_k,
            "query_sample_policy": "uploaded_transient_query_not_inserted_into_historical_db",
        }
        if db_evidence.get("status") != "success":
            evidence.uncertainty_flags.append("database_evidence_unavailable")

        if len(plan.steps) > 4:
            plan.steps[4].status = "completed"
            plan.steps[4].detail = "experimental_db_query_feature_extraction_and_neighbor_retrieval"
            plan.steps[4].output = {
                "sample_id": db_evidence.get("sample_id"),
                "neighbor_count": db_evidence.get("neighbor_count"),
                "top_k": top_k,
            }
        if len(plan.steps) > 5:
            plan.steps[5].status = "completed" if db_evidence.get("status") == "success" else "failed"
            plan.steps[5].detail = "property_evidence_and_expert_reasoning"
        if len(plan.steps) > 6:
            plan.steps[6].status = "completed"
        plan.status = "completed" if db_evidence.get("status") == "success" else "failed"

        answer = self._format_experimental_db_planner_answer(db_evidence)
        conclusion = self._build_conclusion(
            direct_findings={
                "query_workbook": str(sample_path),
                "sample_id": db_evidence.get("sample_id"),
                "neighbor_count": db_evidence.get("neighbor_count"),
                "top_k": top_k,
            },
            database_inference=(db_evidence.get("property_summary", {}) if isinstance(db_evidence, dict) else {}),
            uncertainty={
                "database_evidence_status": db_evidence.get("status"),
                "database_evidence_message": db_evidence.get("message"),
                "query_sample_policy": "uploaded_transient_query_not_inserted_into_historical_db",
            },
            confidence="moderate" if db_evidence.get("status") == "success" else "low",
        )
        self.agent.sessions.remember_tool_result(session_id, "database.experimental_history", {
            "status": db_evidence.get("status"),
            "message": db_evidence.get("message"),
            "evidence": db_evidence,
            "database_evidence": db_evidence,
            "tool_evidence": evidence.tool_evidence,
            "result": {
                "sample_id": db_evidence.get("sample_id"),
                "neighbor_count": db_evidence.get("neighbor_count"),
                "property_summary": db_evidence.get("property_summary", {}),
            },
        })
        self.agent.sessions.set_last_plan(session_id, plan.to_dict())
        self.agent._record_turn(session_id, query, answer)
        return {
            "session_id": session_id,
            "task_type": "planner",
            "used_model": "openai_expert" if isinstance((db_evidence.get("presentation") or {}).get("constrained_reasoning"), dict) else "none",
            "used_tools": ["database.experimental_history"],
            "answer": answer,
            "raw_result": {
                "planner": plan.to_dict(),
                "analysis_request": analysis_request.to_dict(),
                "evidence_bundle": evidence.to_dict(),
                "analysis_conclusion": conclusion.to_dict(),
                "experimental_db_evidence": db_evidence,
            },
        }

    @staticmethod
    def _format_experimental_db_planner_answer(db_evidence: dict[str, Any]) -> str:
        if not isinstance(db_evidence, dict) or db_evidence.get("status") != "success":
            return (
                "Experimental DB evidence inference did not complete.\n"
                f"Status: {db_evidence.get('status') if isinstance(db_evidence, dict) else 'error'}\n"
                f"Message: {db_evidence.get('message') if isinstance(db_evidence, dict) else 'invalid evidence payload'}"
            )
        sample_id = db_evidence.get("sample_id")
        neighbor_count = db_evidence.get("neighbor_count")
        properties = db_evidence.get("property_summary") if isinstance(db_evidence.get("property_summary"), dict) else {}
        lines = [
            "**Planner-routed Experimental DB evidence inference**",
            "",
            f"Query sample: `{sample_id}`.",
            f"Retrieved historical neighbors: {neighbor_count}.",
            "The uploaded workbook was used as a transient query sample and was not inserted into the historical database.",
            "",
            "| Property | DB evidence center | Evidence range | Supporting samples |",
            "|---|---:|---|---|",
        ]
        for name, item in list(properties.items())[:12]:
            if not isinstance(item, dict):
                continue
            estimate = item.get("weighted_estimate")
            evidence_range = item.get("evidence_range")
            if isinstance(evidence_range, list) and len(evidence_range) == 2:
                range_text = f"{evidence_range[0]} to {evidence_range[1]}"
            else:
                range_text = "NA"
            support = item.get("supporting_samples")
            if isinstance(support, list):
                support_text = ", ".join(str(value) for value in support[:3])
                if len(support) > 3:
                    support_text += ", ..."
            else:
                support_text = "NA"
            lines.append(f"| {name} | {estimate} | {range_text} | {support_text} |")
        constrained = (db_evidence.get("presentation") or {}).get("constrained_reasoning") if isinstance(db_evidence.get("presentation"), dict) else {}
        expert = constrained.get("expert_reasoning") if isinstance(constrained, dict) else None
        if isinstance(expert, dict) and expert.get("status") == "success" and expert.get("answer"):
            lines.extend(["", "### Evidence-constrained final inference", str(expert.get("answer"))])
        else:
            statement = (db_evidence.get("presentation") or {}).get("evidence_statement") if isinstance(db_evidence.get("presentation"), dict) else ""
            if statement:
                lines.extend(["", "### Evidence statement", str(statement)])
        return "\n".join(lines)

    def _execute_concept_plan(
        self,
        *,
        plan: ExecutionPlan,
        analysis_request: AnalysisRequest,
        query: str,
        session_id: str,
        retrieval_overrides: dict | None = None,
    ) -> dict:
        capabilities = self._build_capability_schemas(template=plan.template)
        evidence = self._empty_evidence_bundle(capabilities)
        plan.steps[0].status = "completed"
        docs = self._safe_retrieve(query, top_k=2, retrieval_overrides=retrieval_overrides)
        plan.steps[1].status = "completed"
        plan.steps[1].output = {"doc_count": len(docs)}
        evidence.knowledge_evidence = {"documents": docs, "doc_count": len(docs)}
        context = None
        if docs:
            context = "\n\n".join(
                f"[Source: {Path(doc['source']).name} | score={doc['score']:.4f}]\n{doc['text']}"
                for doc in docs
            )
        answer = self.agent.llm.chat(query, context=context)
        plan.steps[2].status = "completed"
        plan.status = "completed"
        conclusion = self._build_conclusion(
            knowledge_interpretation={"answer_mode": "concept_qa", "doc_count": len(docs)},
            uncertainty={},
            confidence="moderate" if docs else "low",
        )
        self.agent.sessions.set_last_plan(session_id, plan.to_dict())
        self.agent._record_turn(session_id, query, answer)
        return {
            "session_id": session_id,
            "task_type": "planner",
            "used_model": self.agent.settings.deepseek_chat_model,
            "used_tools": [],
            "answer": answer,
            "raw_result": {
                "planner": plan.to_dict(),
                "analysis_request": analysis_request.to_dict(),
                "evidence_bundle": evidence.to_dict(),
                "analysis_conclusion": conclusion.to_dict(),
                "docs": docs,
            },
        }

    def _execute_knowledge_plan(
        self,
        *,
        plan: ExecutionPlan,
        analysis_request: AnalysisRequest,
        query: str,
        session_id: str,
        rag_mode_override: str | None = None,
        retrieval_overrides: dict | None = None,
    ) -> dict:
        capabilities = self._build_capability_schemas(template=plan.template, rag_mode_override=rag_mode_override)
        evidence = self._empty_evidence_bundle(capabilities)
        plan.steps[0].status = "completed"
        docs = self._safe_retrieve(query, top_k=self.agent.settings.rag_top_k, retrieval_overrides=retrieval_overrides)
        plan.steps[1].status = "completed"
        plan.steps[1].output = {"doc_count": len(docs)}
        evidence.knowledge_evidence = {"documents": docs, "doc_count": len(docs)}
        rag_result = self._safe_rag_answer(
            query,
            top_k=self.agent.settings.rag_top_k,
            rag_mode_override=rag_mode_override,
            retrieval_overrides=retrieval_overrides,
        )
        rag_mode = (
            rag_mode_override
            or ("qa_oriented" if bool(getattr(self.agent.settings, "advanced_rag_enabled", False)) else "naive")
        ).strip().lower()
        structured_rag = build_structured_rag_evidence(
            rag_result.get("raw_result"),
            mode=rag_mode,
            fallback_docs=docs,
        )
        plan.steps[2].status = "completed"
        plan.steps[3].status = "completed"
        plan.status = "completed"
        self.agent.sessions.set_last_plan(session_id, plan.to_dict())
        answer = rag_result.get("answer", "")
        if not answer:
            context = None
            if docs:
                context = "\n\n".join(
                    f"[Source: {Path(doc['source']).name} | score={doc['score']:.4f}]\n{doc['text']}"
                    for doc in docs
                )
            answer = self.agent.llm.chat(query, context=context)
        evidence.knowledge_evidence = structured_rag.get("knowledge_evidence", {})
        evidence.graph_evidence = structured_rag.get("graph_evidence", {})
        conclusion = self._build_conclusion(
            knowledge_interpretation={
                "answer_mode": "knowledge_qa",
                "doc_count": structured_rag.get("knowledge_evidence", {}).get("doc_count", len(docs)),
                "rag_error": rag_result.get("error"),
                "rag_mode": rag_mode,
            },
            uncertainty={"retrieve_error": rag_result.get("error")},
            confidence="moderate" if docs else "low",
        )
        self.agent._record_turn(session_id, query, answer)
        return {
            "session_id": session_id,
            "task_type": "planner",
            "used_model": rag_result.get("used_model", self.agent.settings.deepseek_chat_model),
            "used_tools": [],
            "answer": answer,
            "raw_result": {
                "planner": plan.to_dict(),
                "analysis_request": analysis_request.to_dict(),
                "evidence_bundle": evidence.to_dict(),
                "analysis_conclusion": conclusion.to_dict(),
                "docs": rag_result.get("raw_result", docs),
                "retrieve_error": rag_result.get("error"),
            },
        }

    def _execute_processing_plan(
        self,
        *,
        plan: ExecutionPlan,
        analysis_request: AnalysisRequest,
        query: str,
        session_id: str,
        task_type_override: str | None = None,
        retrieval_overrides: dict | None = None,
    ) -> dict:
        capabilities = self._build_capability_schemas(template=plan.template)
        evidence = self._empty_evidence_bundle(capabilities)
        plan.steps[0].status = "completed"
        extract_prompt = render_prompt("planner.processing.extract", query=query)
        extracted = self.agent.llm.chat(extract_prompt)
        plan.steps[1].status = "completed"
        plan.steps[1].output = {"extracted": extracted}
        docs = self._safe_retrieve(query, top_k=self.agent.settings.rag_top_k, retrieval_overrides=retrieval_overrides)
        plan.steps[2].status = "completed"
        plan.steps[2].output = {"doc_count": len(docs)}
        evidence.knowledge_evidence = {"documents": docs, "doc_count": len(docs)}
        context = "\n\n".join(
            f"[Source: {Path(doc['source']).name} | score={doc['score']:.4f}]\n{doc['text']}"
            for doc in docs
        )
        reasoning_prompt = render_prompt(
            "planner.processing.reason",
            query=query,
            extracted=extracted,
            context=context,
        )
        answer = self.agent.llm.reason(reasoning_prompt)
        plan.steps[3].status = "completed"
        plan.steps[4].status = "completed"
        plan.status = "completed"
        conclusion = self._build_conclusion(
            direct_findings={"extracted_parameters": extracted},
            knowledge_interpretation={"doc_count": len(docs), "mode": "processing_evaluation"},
            uncertainty={},
            confidence="moderate" if docs else "low",
        )
        self.agent.sessions.set_last_plan(session_id, plan.to_dict())
        self.agent._record_turn(session_id, query, answer)
        return {
            "session_id": session_id,
            "task_type": "planner",
            "used_model": self.agent.settings.deepseek_reasoner_model,
            "used_tools": [],
            "answer": answer,
            "raw_result": {
                "planner": plan.to_dict(),
                "analysis_request": analysis_request.to_dict(),
                "evidence_bundle": evidence.to_dict(),
                "analysis_conclusion": conclusion.to_dict(),
                "docs": docs,
                "extracted": extracted,
            },
        }
