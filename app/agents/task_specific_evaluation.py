from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


EvaluationTarget = Literal["tool"]


@dataclass(slots=True)
class TaskSpecificEvaluationCase:
    """Minimal offline evaluation case used by tool onboarding/provider review."""

    case_id: str
    evaluation_target: EvaluationTarget
    query: str
    expected_route: str | None = None
    expected_template: str | None = None
    expected_capabilities: list[str] = field(default_factory=list)
    expected_evidence: list[str] = field(default_factory=list)
    expected_conclusion_fields: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaskSpecificEvaluationResult:
    case_id: str
    evaluation_target: EvaluationTarget
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    actual: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def task_specific_case_from_dict(payload: dict[str, Any]) -> TaskSpecificEvaluationCase:
    """Read a generated or hand-authored tool-provider evaluation case."""

    return TaskSpecificEvaluationCase(
        case_id=str(payload.get("case_id") or "unnamed_case"),
        evaluation_target="tool",
        query=str(payload.get("query") or ""),
        expected_route=payload.get("expected_route"),
        expected_template=payload.get("expected_template"),
        expected_capabilities=_string_list(payload.get("expected_capabilities")),
        expected_evidence=_string_list(payload.get("expected_evidence")),
        expected_conclusion_fields=_string_list(payload.get("expected_conclusion_fields")),
        notes=str(payload.get("notes") or ""),
    )


def task_specific_cases_from_payload(payload: Any) -> list[TaskSpecificEvaluationCase]:
    if not isinstance(payload, list):
        return []
    return [task_specific_case_from_dict(item) for item in payload if isinstance(item, dict)]


def evaluate_tool_provider_result(
    case: TaskSpecificEvaluationCase,
    provider_evaluation: dict[str, Any],
) -> TaskSpecificEvaluationResult:
    """Evaluate a reviewed Tool Onboarding provider against one tool case."""

    capability = provider_evaluation.get("capability") if isinstance(provider_evaluation, dict) else {}
    capability = capability if isinstance(capability, dict) else {}
    manifest = provider_evaluation.get("manifest") if isinstance(provider_evaluation, dict) else {}
    manifest = manifest if isinstance(manifest, dict) else {}
    execution_result = provider_evaluation.get("execution_result") if isinstance(provider_evaluation, dict) else None
    evidence_contract = manifest.get("evidence_contract") if isinstance(manifest.get("evidence_contract"), dict) else {}
    declared_outputs = _string_list(evidence_contract.get("required_outputs"))
    executed = isinstance(execution_result, dict) and execution_result.get("status") != "skipped"

    capability_id = capability.get("capability_id") or manifest.get("capability_id")
    checks = {
        "evaluation_target": case.evaluation_target == "tool",
        "provider_package": bool(provider_evaluation.get("passed")),
        "capabilities": all(expected == capability_id for expected in case.expected_capabilities),
        "declared_evidence": all(output in declared_outputs for output in case.expected_evidence),
        "execution_evidence": _contains_provider_outputs(execution_result, case.expected_evidence) if executed else True,
    }
    failure_reasons = [name for name, ok in checks.items() if not ok]
    return TaskSpecificEvaluationResult(
        case_id=case.case_id,
        evaluation_target=case.evaluation_target,
        passed=all(checks.values()),
        checks=checks,
        actual={
            "capability_id": capability_id,
            "declared_outputs": declared_outputs,
            "executed": executed,
            "provider_passed": bool(provider_evaluation.get("passed")),
        },
        expected=case.to_dict(),
        failure_reasons=failure_reasons,
    )


def summarize_task_specific_results(results: list[TaskSpecificEvaluationResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for item in results if item.passed)
    failed = total - passed
    failure_counter = Counter(reason for item in results for reason in item.failure_reasons)
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": (passed / total) if total else 0.0,
        "failure_reasons": dict(sorted(failure_counter.items())),
    }


def _contains_provider_outputs(execution_result: Any, expected_outputs: list[str]) -> bool:
    if not expected_outputs:
        return True
    if not isinstance(execution_result, dict):
        return False

    candidates = set()
    tool_evidence = execution_result.get("tool_evidence")
    if isinstance(tool_evidence, dict):
        candidates.update(tool_evidence.keys())
        if tool_evidence.get("structured_result"):
            candidates.add("tool_evidence")
            candidates.add("structured_tool_evidence")
    result_payload = execution_result.get("result")
    if isinstance(result_payload, dict):
        candidates.update(result_payload.keys())
    artifacts = execution_result.get("artifacts")
    if isinstance(artifacts, dict) and artifacts:
        candidates.add("artifacts")

    normalized_candidates = {str(item) for item in candidates}
    return all(output in normalized_candidates or output == "tool_evidence" for output in expected_outputs)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []
