from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.agents.state_store import SessionArtifact, SessionMemoryStore
from app.config import get_settings
from app.llm.deepseek_client import DeepSeekClient
from app.llm.openai_expert_client import OpenAIExpertClient
from app.rag.kb_backend import (
    iterative_review_rag_answer,
    naive_rag_answer,
    qa_oriented_rag_answer,
    retrieve,
)
from app.tools.experimental_db import SampleDatabase
from app.tools.experimental_db.evidence_retriever import attach_petroleum_expert_reasoning
from app.tools.experimental_db.petroleum_fraction_spectral_evidence import (
    build_petroleum_fraction_spectral_workbook_evidence,
)
from app.tools.registry import ToolRegistry, ToolSpec
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
session_store = SessionMemoryStore()


class PetroAnalystAgent:
    """Minimal runtime execution layer for the public PetroAnalyst release."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = DeepSeekClient()
        self.sessions = session_store
        self.registry = ToolRegistry()
        self._register_tools()

    def _register_tools(self) -> None:
        template_path = (
            Path(__file__).resolve().parents[2]
            / 'data'
            / 'experimental_db'
            / 'templates'
        / 'petroanalyst_experimental_sample_template.xlsx'
        ).resolve()
        self.registry.register(
            'database.experimental_history',
            self._run_experimental_history_workbook_tool,
            spec=ToolSpec(
                name='database.experimental_history',
                description=(
                    'Historical Experimental Database evidence path for uploaded petroleum fraction query workbooks. '
                    'It extracts GC-FID and IR features, retrieves top-k historical neighbors, '
                    'and returns property evidence for training-free inference.'
                ),
                parameters={
                    'type': 'object',
                    'properties': {
                        'input_file': {'type': 'string', 'description': 'Uploaded petroleum fraction sample workbook.'},
                        'query': {'type': 'string', 'description': 'Property inference request.'},
                        'top_k': {'type': 'integer', 'default': 10},
                    },
                    'required': ['input_file'],
                    'x_petroanalyst_input_requirements': {
                        'file_type': '.xlsx Excel workbook',
                        'required_workbook_sheets': ['sample_information', 'gc_fid', 'ir_spectrum'],
                        'notes': [
                            'The query workbook is treated as a transient sample and is not inserted into the historical database.',
                            'GC-FID and IR spectra are mapped to the saved experimental-database feature space before neighbor retrieval.',
                        ],
                        'input_templates': [str(template_path)],
                    },
                },
                requires_upload=True,
                accepted_file_types=('.xlsx', '.xls'),
                reusable_from_previous_file=True,
                intent_keywords=(
                    'experimental database',
                    'historical database',
                    'historical experimental database',
                    'historical samples',
                    'property inference',
                    'training-free inference',
                    'density',
                    'boiling point',
                    'distillation',
                    'saturates',
                    'gc-fid',
                    'ir',
                    'spectral feature',
                    'petroleum fraction',
                ),
            ),
        )

    def _build_upload_instruction(self, tool_spec: ToolSpec) -> str:
        if not isinstance(tool_spec.parameters, dict):
            return f'Upload the required workbook or data file for `{tool_spec.name}`.'
        requirements = tool_spec.parameters.get('x_petroanalyst_input_requirements')
        requirements = requirements if isinstance(requirements, dict) else {}
        lines = [f'Upload the required input file for `{tool_spec.name}`, then rerun the analytical-tool request.']
        file_type = requirements.get('file_type')
        if file_type:
            lines.append(f'Required file type: {file_type}.')
        required_sheets = requirements.get('required_sheets') or requirements.get('required_workbook_sheets')
        if isinstance(required_sheets, dict) and required_sheets:
            sheet_lines: list[str] = []
            for sheet_name, columns in required_sheets.items():
                if isinstance(columns, list):
                    sheet_lines.append(f"{sheet_name} ({', '.join(str(col) for col in columns)})")
                else:
                    sheet_lines.append(str(sheet_name))
            lines.append('Required sheets: ' + '; '.join(sheet_lines) + '.')
        elif isinstance(required_sheets, list) and required_sheets:
            lines.append('Required sheets: ' + ', '.join(str(item) for item in required_sheets) + '.')
        return ' '.join(lines)

    @staticmethod
    def _input_template_file_from_spec(tool_spec: ToolSpec) -> str | None:
        if not isinstance(tool_spec.parameters, dict):
            return None
        requirements = tool_spec.parameters.get('x_petroanalyst_input_requirements')
        if not isinstance(requirements, dict):
            return None
        templates = requirements.get('input_templates')
        if not isinstance(templates, list):
            return None
        for template in templates:
            if isinstance(template, str) and template.strip():
                return template.strip()
            if isinstance(template, dict):
                candidate = str(template.get('path') or '').strip()
                if candidate:
                    return candidate
        return None

    def _build_upload_spec_with_assets(self, tool_name: str, upload_spec: dict | None) -> dict[str, Any]:
        spec = dict(upload_spec or {})
        spec.setdefault('status', 'awaiting_upload')
        spec.setdefault('tool_name', tool_name)
        tool_spec = self.registry.get_spec(tool_name)
        if tool_spec:
            spec.setdefault('accepted_file_types', list(tool_spec.accepted_file_types))
            spec.setdefault('instructions', self._build_upload_instruction(tool_spec))
            template_file = self._input_template_file_from_spec(tool_spec)
            if template_file:
                spec.setdefault('template_input_file', template_file)
        else:
            spec.setdefault('accepted_file_types', ['.xlsx', '.xls', '.csv', '.json', '.txt'])
            spec.setdefault('instructions', f'Upload the required workbook or data file for `{tool_name}`.')
        return spec

    def _record_turn(self, session_id: str, user_query: str, answer: str) -> None:
        self.sessions.append_message(session_id, 'user', user_query)
        self.sessions.append_message(session_id, 'assistant', answer)

    def _should_reference_last_tool(self, query: str, session_id: str) -> bool:
        session = self.sessions.get_or_create(session_id)
        if not session.last_tool_result:
            return False
        lowered = str(query or '').lower()
        markers = (
            'result',
            'tool result',
            'experimental database',
            'historical neighbor',
            'historical sample',
            'property evidence',
            'density',
            'distillation',
            'boiling point',
            'saturates',
            'reasoning',
            'interpret',
            'explain',
            'compare',
            'summarize',
        )
        return any(marker in lowered for marker in markers)

    def _build_context(self, session_id: str, query: str) -> str | None:
        parts: list[str] = []
        recent = self.sessions.build_recent_context(session_id)
        if recent:
            parts.append(recent)
        task_state = self.sessions.build_task_state_context(session_id)
        if task_state:
            parts.append(task_state)
        if self._should_reference_last_tool(query, session_id):
            last_tool = self.sessions.build_last_tool_context(session_id)
            if last_tool:
                parts.append(last_tool)
        return '\n\n'.join(part for part in parts if part) or None

    def _requires_compact_answer(self, query: str) -> bool:
        lowered = str(query or '').lower()
        narrow = ('how many', 'content', 'fraction', 'ratio', 'highest', 'largest', 'lowest', 'smallest', 'top ', 'list', 'table')
        expansive = ('analyze', 'analysis', 'evaluate', 'recommend', 'processing route', 'meaning', 'reason', 'why')
        return any(marker in lowered for marker in narrow) and not any(marker in lowered for marker in expansive)

    def _build_response_query(self, query: str) -> str:
        if self._requires_compact_answer(query):
            return (
                'Answer only with the data or conclusion directly required by the current question. '
                'Do not expand into background discussion, processing suggestions, or long explanations.\n\n'
                f'User question: {query}'
            )
        return query

    def handle_uploaded_tool(self, *, tool_name: str, payload: dict, session_id: str | None, user_query: str) -> dict[str, Any]:
        session = self.sessions.get_or_create(session_id)

        if tool_name == 'database.experimental_history':
            file_path = str(payload.get('input_file') or payload.get('file_path') or payload.get('excel_file') or '').strip()
            if not file_path:
                answer = 'Upload a petroleum fraction query workbook before running Experimental Database evidence inference.'
                return {
                    'status': 'awaiting_upload',
                    'result': self._build_upload_spec_with_assets(tool_name, {'instructions': answer}),
                    'message': 'input_workbook_required',
                    'session_id': session.session_id,
                    'answer': answer,
                }
            input_path = Path(file_path).resolve()
            input_artifact = SessionArtifact(
                path=str(input_path),
                file_name=input_path.name,
                file_type=input_path.suffix.lower(),
                source_tool=tool_name,
                role='input',
            )
            if not any(item.path == input_artifact.path for item in session.artifacts):
                session.artifacts.append(input_artifact)
                session.artifacts = session.artifacts[-12:]
            query_text = user_query or str(payload.get('query') or '')
            tool_result = self._run_experimental_history_workbook_tool(
                {
                    'input_file': str(input_path),
                    'file_path': str(input_path),
                    'excel_file': str(input_path),
                    'query': query_text,
                    'top_k': self._parse_top_k(query_text, default=10),
                }
            )
            self.sessions.remember_tool_result(session.session_id, tool_name, tool_result)
            evidence = tool_result.get('evidence') if isinstance(tool_result, dict) else {}
            property_summary = evidence.get('property_summary') if isinstance(evidence, dict) else {}
            answer = f'{tool_name} completed.'
            if isinstance(evidence, dict):
                answer += f" Retrieved {evidence.get('neighbor_count', 0)} historical neighbors."
            if isinstance(property_summary, dict) and property_summary:
                answer += f" Property evidence fields: {', '.join(list(property_summary)[:8])}."
            result_payload = {
                'evidence': evidence,
                'tool_evidence': tool_result.get('tool_evidence', {}),
                'database_evidence': evidence,
                'input_file': str(input_path),
            }
            self._record_turn(session.session_id, user_query, answer)
            return {
                'status': tool_result.get('status', 'success'),
                'result': result_payload,
                'message': tool_result.get('message', ''),
                'session_id': session.session_id,
                'answer': answer,
            }

        if tool_name.startswith('tool.') and self.registry.get(tool_name) is not None:
            file_path = str(payload.get('input_file') or payload.get('file_path') or payload.get('excel_file') or '').strip()
            if not file_path:
                answer = 'Upload an input workbook or data file before running this analytical tool.'
                return {
                    'status': 'awaiting_upload',
                    'result': self._build_upload_spec_with_assets(tool_name, {'instructions': answer}),
                    'message': 'input_file_required',
                    'session_id': session.session_id,
                    'answer': answer,
                }
            input_path = Path(file_path).resolve()
            input_artifact = SessionArtifact(
                path=str(input_path),
                file_name=input_path.name,
                file_type=input_path.suffix.lower(),
                source_tool=tool_name,
                role='input',
            )
            if not any(item.path == input_artifact.path for item in session.artifacts):
                session.artifacts.append(input_artifact)
                session.artifacts = session.artifacts[-12:]
            output_dir = payload.get('output_dir') or str(input_path.parent / f'{input_path.stem}_{tool_name}')
            tool_result = self.registry.call(
                tool_name,
                {
                    'input_file': str(input_path),
                    'file_path': str(input_path),
                    'excel_file': str(input_path),
                    'output_dir': str(output_dir),
                },
            )
            self.sessions.remember_tool_result(session.session_id, tool_name, tool_result)
            tool_evidence = tool_result.get('tool_evidence') if isinstance(tool_result, dict) else {}
            structured = tool_evidence.get('structured_result') if isinstance(tool_evidence, dict) else None
            artifacts = tool_result.get('artifacts') if isinstance(tool_result, dict) else {}
            answer = f'{tool_name} completed.'
            if isinstance(structured, list):
                answer += f' Structured evidence rows: {len(structured)}.'
            elif isinstance(structured, dict) and structured:
                answer += f' Structured evidence fields: {len(structured)}.'
            if isinstance(artifacts, dict) and artifacts:
                answer += ' Generated artifacts are available in the tool result.'
            result_payload = tool_result.get('result')
            if isinstance(result_payload, dict):
                result_payload = dict(result_payload)
            else:
                result_payload = {}
            result_payload.setdefault('tool_evidence', tool_evidence if isinstance(tool_evidence, dict) else {})
            result_payload.setdefault('artifacts', artifacts if isinstance(artifacts, dict) else {})
            if isinstance(tool_result, dict):
                for artifact_key in ('output_excel', 'output_file', 'chart_html'):
                    if tool_result.get(artifact_key) and artifact_key not in result_payload:
                        result_payload[artifact_key] = tool_result[artifact_key]
            self._record_turn(session.session_id, user_query, answer)
            return {
                'status': tool_result.get('status', 'success'),
                'result': result_payload,
                'message': tool_result.get('message', ''),
                'session_id': session.session_id,
                'answer': answer,
            }

        return {
            'status': 'error',
            'result': None,
            'message': f'Upload handling is not available for `{tool_name}` in the release version.',
            'session_id': session.session_id,
        }

    def _run_experimental_history_workbook_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        input_file = str(payload.get('input_file') or payload.get('file_path') or payload.get('excel_file') or '').strip()
        if not input_file:
            return {'status': 'awaiting_upload', 'message': 'input_workbook_required', 'result': None}
        workbook = Path(input_file).resolve()
        query = str(
            payload.get('query')
            or 'Use IR and GC-FID to infer density, distillation distribution, and saturates for this petroleum fraction sample.'
        )
        top_k = int(payload.get('top_k') or self._parse_top_k(query, default=10))
        workspace = Path(__file__).resolve().parents[2] / 'data' / 'experimental_db'
        evidence = build_petroleum_fraction_spectral_workbook_evidence(
            workbook=workbook,
            template_definition=SampleDatabase(workspace).get_template_definition(),
            query=query,
            top_k=top_k,
        )
        evidence = attach_petroleum_expert_reasoning(evidence, OpenAIExpertClient())
        status = str(evidence.get('status') or 'success') if isinstance(evidence, dict) else 'error'
        return {
            'status': status,
            'message': str(evidence.get('message') or 'experimental_database_evidence_ready') if isinstance(evidence, dict) else 'experimental_database_evidence_failed',
            'sample_id': evidence.get('sample_id') if isinstance(evidence, dict) else workbook.stem,
            'evidence': evidence,
            'database_evidence': evidence,
            'tool_evidence': {
                'capability_id': 'database.experimental_history',
                'status': status,
                'input_file': str(workbook),
                'top_k': top_k,
                'structured_result': evidence.get('property_summary', {}) if isinstance(evidence, dict) else {},
            },
            'result': {
                'sample_id': evidence.get('sample_id') if isinstance(evidence, dict) else workbook.stem,
                'neighbor_count': evidence.get('neighbor_count') if isinstance(evidence, dict) else None,
                'property_summary': evidence.get('property_summary', {}) if isinstance(evidence, dict) else {},
            },
        }

    @staticmethod
    def _parse_top_k(query: str, *, default: int = 10) -> int:
        match = re.search(r'(?:top[-_\s]*k|k)\s*[=:]?\s*(\d+)', str(query or ''), flags=re.I)
        if not match:
            return max(1, min(50, int(default)))
        return max(1, min(50, int(match.group(1))))

    def answer_query(
        self,
        query: str,
        task_type_override: str | None = None,
        session_id: str | None = None,
        rag_mode_override: str | None = None,
        retrieval_overrides: dict | None = None,
    ) -> dict[str, Any]:
        session = self.sessions.get_or_create(session_id)
        override = (task_type_override or '').strip().lower() or 'chat'
        if override == 'auto':
            override = 'chat'

        if override == 'tool':
            answer = (
                'Direct tool orchestration is no longer handled by `PetroAnalystAgent.answer_query()` '
                'in the release version. Use `PlannerOrchestrator` or `/tools/upload` instead.'
            )
            self._record_turn(session.session_id, query, answer)
            return {
                'session_id': session.session_id,
                'task_type': 'tool',
                'used_model': 'none',
                'used_tools': [],
                'answer': answer,
                'raw_result': {'status': 'delegated_to_planner'},
            }

        if override == 'rag':
            rag_mode = (rag_mode_override or 'auto').strip().lower()
            docs: list[dict[str, Any]] = []
            use_naive = rag_mode == 'naive' or (
                rag_mode == 'auto' and not bool(getattr(self.settings, 'advanced_rag_enabled', False))
            )
            if use_naive:
                docs = retrieve(query, top_k=self.settings.rag_top_k, retrieval_overrides=retrieval_overrides)
                if not docs:
                    answer = 'No relevant knowledge chunks were retrieved. Please ingest or rebuild the knowledge base first.'
                    self._record_turn(session.session_id, query, answer)
                    return {
                        'session_id': session.session_id,
                        'task_type': 'rag',
                        'used_model': self.settings.deepseek_chat_model,
                        'used_tools': [],
                        'answer': answer,
                        'raw_result': [],
                    }
            if use_naive:
                rag_result = naive_rag_answer(
                    query,
                    top_k=self.settings.rag_top_k,
                    retrieval_overrides=retrieval_overrides,
                )
            elif rag_mode == 'iterative_review':
                rag_result = iterative_review_rag_answer(query, retrieval_overrides=retrieval_overrides)
            else:
                rag_result = qa_oriented_rag_answer(query, retrieval_overrides=retrieval_overrides)
            answer = rag_result.get('answer', '')
            self._record_turn(session.session_id, query, answer)
            return {
                'session_id': session.session_id,
                'task_type': 'rag',
                'used_model': rag_result.get('used_model', self.settings.deepseek_chat_model),
                'used_tools': [],
                'answer': answer,
                'raw_result': rag_result.get('raw_result', docs),
            }

        if override == 'reasoning':
            context = self._build_context(session.session_id, query)
            scoped_query = self._build_response_query(query)
            reason_query = scoped_query if not context else f'{context}\n\nCurrent question: {scoped_query}'
            answer = self.llm.reason(reason_query)
            self._record_turn(session.session_id, query, answer)
            return {
                'session_id': session.session_id,
                'task_type': 'reasoning',
                'used_model': self.settings.deepseek_reasoner_model,
                'used_tools': [],
                'answer': answer,
                'raw_result': None,
            }

        context = self._build_context(session.session_id, query)
        answer = self.llm.chat(self._build_response_query(query), context=context)
        self._record_turn(session.session_id, query, answer)
        return {
            'session_id': session.session_id,
            'task_type': 'chat',
            'used_model': self.settings.deepseek_chat_model,
            'used_tools': [],
            'answer': answer,
            'raw_result': None,
        }
