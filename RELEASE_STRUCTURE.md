# PetroAnalyst Release Structure

This document summarizes the public-release layout of `PetroAnalyst`, the role of each main directory, and the main runtime relationships between modules.

## 1. Top-level layout

```text
PetroAnalyst/
- app/                FastAPI application, planner, RAG, tools, and support-layer pages
- data/               Demo Experimental Database assets and related runtime resources
- petroleum_kb/       Literature KB package, retrieval core, demo index assets, and QA datasets
- static_demo/        No-install preview pages for the support layer and keyword graph
- .env.example        Example environment configuration
- README.md           User-facing overview and startup guide
- RELEASE_STRUCTURE.md  Release-structure and dependency guide
- requirements.txt    Unified runtime dependencies
```

## 2. Main application package: `app/`

### 2.1 Root application files

- `app/main.py`
  - FastAPI entrypoint.
  - Registers support-layer pages and backend APIs.
  - Loads the runtime agent and the reviewed capability provider at startup.
- `app/config.py`
  - Central settings model.
  - Resolves runtime directories and environment flags.
- `app/schemas.py`
  - Pydantic request/response schemas used by API routes.
- `app/benchmark_workbench.py`
  - Backend logic for staged KB construction and QA benchmark generation.
- `app/support_layer_workbench.py`
  - Backend helper layer for the RAG and Experimental Database workbenches.

### 2.2 Agent layer: `app/agents/`

- `app/agents/runtime_agent.py`
  - Core PetroAnalyst runtime agent.
  - Holds the tool registry and basic execution logic.
- `app/agents/planner_orchestrator.py`
  - Main orchestration path for planner queries.
  - Selects whether a request should use literature RAG, Experimental Database inference, tool execution, or a follow-up hybrid route.
- `app/agents/state_store.py`
  - Minimal in-memory session store for conversation state, uploaded artifacts, and last tool results.
- `app/agents/schemas.py`
  - Internal agent-side data structures.
- `app/agents/task_specific_evaluation.py`
  - Minimal provider-evaluation helper retained for tool onboarding and reviewed provider packaging.

### 2.3 Capability packaging layer: `app/capabilities/`

- `app/capabilities/provider_registry.py`
  - Reads, validates, saves, reviews, enables, and revokes reviewed capability-provider packages.
  - Registers reviewed tools into the runtime registry on startup.

#### Released reviewed provider

- `app/capabilities/providers/hydrocarbon_type_calculator/capability.json`
  - Capability-level schema for the released hydrocarbon-type tool.
- `app/capabilities/providers/hydrocarbon_type_calculator/manifest.json`
  - Provider manifest describing runtime, artifacts, and templates.
- `app/capabilities/providers/hydrocarbon_type_calculator/package.json`
  - Packaged reviewed provider definition loaded by the registry.
- `app/capabilities/providers/hydrocarbon_type_calculator/evaluation_cases.json`
  - Provider-level evaluation cases.
- `app/capabilities/providers/hydrocarbon_type_calculator/registration_plan.json`
  - Registration metadata used by the onboarding flow.
- `app/capabilities/providers/hydrocarbon_type_calculator/sht0606_official_full.py`
  - Released hydrocarbon-type calculator script.
- `app/capabilities/providers/hydrocarbon_type_calculator/templates/input_template.xlsx`
  - Input workbook template for the hydrocarbon-type calculator.

### 2.4 LLM client layer: `app/llm/`

- `app/llm/deepseek_client.py`
  - Base LLM client used for chat, RAG generation, and planner prompting.
- `app/llm/openai_expert_client.py`
  - Role-defined expert LLM client used for Experimental Database evidence interpretation.

### 2.5 RAG layer: `app/rag/`

- `app/rag/kb_backend.py`
  - Adapter between the application and `petroleum_kb`.
  - Exposes the three released RAG routes: `naive`, `qa_oriented`, and `iterative_review`.
- `app/rag/agentic_rag.py`
  - Implements QA-oriented RAG and iterative-review RAG loops.
- `app/rag/evidence_filter.py`
  - LLM-based evidence filtering module used by advanced RAG routes.
- `app/rag/query_refinement_agent.py`
  - Query-refinement logic for retrieval improvement.
- `app/rag/run_log.py`
  - Runtime log helper for RAG tracing.
- `app/rag/chunking.py`
  - Local chunking helpers retained for internal RAG support.

### 2.6 Frontend static files: `app/static/`

- `app/static/support_layer.html`
  - Main support-layer landing page.
- `app/static/benchmark_workbench.html`
  - KB Construction and QA Generation page.
- `app/static/rag_workbench.html`
  - PetroAnalyst RAG workbench page.
- `app/static/experimental_db_workbench.html`
  - Experimental Database inference workbench page.
- `app/static/experimental_db_manager.html`
  - Experimental Database browsing and management page.
- `app/static/tool_onboarding.html`
  - Analytical Tool Onboarding page.
- `app/static/support_workbench.css`
  - Shared stylesheet.
- `app/static/support_help.js`
  - Shared help-modal logic.
- `app/static/support_manual.js`
  - Shared manual/help interaction logic.
- `app/static/support_benchmark.js`
  - Frontend logic for KB and QA generation.
- `app/static/support_rag.js`
  - Frontend logic for RAG testing.
- `app/static/support_experimental_db.js`
  - Frontend logic for Experimental Database testing and import.
- `app/static/experimental_db_manager.js`
  - Frontend logic for stored-sample browsing and deletion.
- `app/static/support_tool_onboarding.js`
  - Frontend logic for provider packaging, review, and runtime testing.

### 2.7 Tool layer: `app/tools/`

- `app/tools/registry.py`
  - Tool and capability registry abstraction.
- `app/tools/tool_onboarding.py`
  - Converts reviewed scripts or callables into packaged capabilities.
- `app/tools/adapters/generic_external_python_adapter.py`
  - Generic adapter for invoking reviewed Python-based tool providers.

#### Experimental Database support code

- `app/tools/experimental_db/sample_database.py`
  - Sample-centered historical Experimental Database implementation.
  - Handles sample import, storage, catalog updates, and template handling.
- `app/tools/experimental_db/evidence_retriever.py`
  - Builds historical-neighbor evidence from stored sample records.
- `app/tools/experimental_db/petroleum_fraction_spectral_evidence.py`
  - Manuscript-aligned training-free property-inference path.
  - Projects GC-FID and IR workbook data into saved PCA/PLS latent space and retrieves historical neighbors.

### 2.8 Utilities: `app/utils/`

- `app/utils/logger.py`
  - Shared logging setup.

## 3. Experimental Database assets: `data/experimental_db/`

### 3.1 Templates

- `data/experimental_db/templates/petroanalyst_experimental_sample_template.xlsx`
  - Main workbook template for Experimental Database import and query inference.
- `data/experimental_db/templates/sample_import.json`
  - Example import payload structure.
- `data/experimental_db/templates/template_definition.json`
  - Active workbook parsing definition used by `sample_database.py`.

### 3.2 Demo stored samples

- `data/experimental_db/samples/`
  - Demo stored historical sample records for public-release browsing and basic DB inspection.
- `data/experimental_db/catalog/experimental_samples.sqlite`
  - SQLite catalog used by the Experimental Database manager and sample index.
- `data/experimental_db/registry/`
  - JSON/JSONL registry files for stored sample information and artifacts.

### 3.3 Historical-neighbor feature package

Directory:

- `data/experimental_db/feature_models/petroleum_fraction_demo_20260712/`

Files:

- `feature_matrix_all.csv`
  - Demo latent-feature matrix for historical and query/test samples.
- `feature_target_correlations.csv`
  - Correlation-derived feature weights for target-property relevance.
- `property_table_all.csv`
  - Demo property table for historical/test samples.
- `sample_metadata.csv`
  - Sample metadata, including train/test split information.
- `spectral_models/model_manifest.json`
  - Manifest listing the retained PCA/PLS model artifacts.
- `spectral_models/*.npz`
  - Saved GC-FID and IR PCA/PLS models used by the workbook-based training-free inference route.

## 4. Literature KB package: `petroleum_kb/`

### 4.1 Wrapper-level files

- `petroleum_kb/config.py`
  - Project-level configuration for the standalone literature KB package.
- `petroleum_kb/main.py`
  - Standalone KB command helper retained from the KB package.
- `petroleum_kb/check_grobid.py`
  - GROBID availability check helper.
- `petroleum_kb/docker-compose.grobid.yml`
  - Optional local GROBID deployment file.

### 4.2 QA benchmark generation helpers

- `petroleum_kb/eval/build_dataset.py`
  - Main benchmark-construction script invoked by `app/benchmark_workbench.py`.
- `petroleum_kb/eval/dataset_generation_utils.py`
  - Shared dataset-generation helpers used by `build_dataset.py`.
- `petroleum_kb/eval/datasets/`
  - Released six QA benchmark files used in the manuscript evaluation.

### 4.3 Core KB package: `petroleum_kb/kb_core/`

- `petroleum_kb/kb_core/data/processed/`
  - Demo processed chunk files retained for released RAG testing.
- `petroleum_kb/kb_core/data/index/`
  - Demo FAISS/BM25/index metadata assets retained for released RAG testing.
- `petroleum_kb/kb_core/data/analysis/keyword_graph_text_clean/`
  - Released keyword-graph assets derived from the 977-paper literature corpus.
- `petroleum_kb/kb_core/io/`
  - Loaders and chunking logic.
- `petroleum_kb/kb_core/indexing/`
  - Embedding and vector-store handling.
- `petroleum_kb/kb_core/labeling/`
  - Chunk-labeling helpers.
- `petroleum_kb/kb_core/pipeline/`
  - KB ingest and index-construction pipeline.
- `petroleum_kb/kb_core/retrieval/`
  - Searcher, BM25, reranker, and evidence representations.
- `petroleum_kb/tools/keyword_graph_from_text.py`
  - Used by the KB workbench to build staged keyword graphs from processed chunks.

## 5. Static preview assets: `static_demo/`

- `static_demo/support_layer_demo.html`
  - Standalone no-install support-layer preview page.
- `static_demo/topic_keyword_graph_demo.html`
  - Quick-access copy of the released topic keyword graph.
- `static_demo/support_workbench.css`
  - Shared stylesheet for the static preview page.

## 6. Main runtime relationships

### 6.1 Application entry

1. `uvicorn app.main:app`
2. `app/main.py`
3. `PetroAnalystAgent` is created from `app/agents/runtime_agent.py`
4. `PlannerOrchestrator` is created from `app/agents/planner_orchestrator.py`
5. `provider_registry.py` loads and registers the reviewed hydrocarbon-type provider

### 6.2 Support-layer page routing

1. Browser opens `app/static/support_layer.html`
2. The landing page links to the four published sub-pages:
   - `benchmark_workbench.html`
   - `rag_workbench.html`
   - `experimental_db_workbench.html`
   - `tool_onboarding.html`
3. Each page calls matching API routes in `app/main.py`

### 6.3 RAG route

1. Request enters `app/main.py`
2. `planner_orchestrator.py` selects one of the three released RAG paths:
   - `naive`
   - `qa_oriented`
   - `iterative_review`
3. `app/rag/kb_backend.py` builds the active searcher from `petroleum_kb`
4. Retrieved chunks are processed by `agentic_rag.py`, `evidence_filter.py`, and `query_refinement_agent.py`
5. Final answer returns through `runtime_agent.py` or `planner_orchestrator.py`

### 6.4 KB construction and QA generation route

1. `benchmark_workbench.html` calls backend routes in `app/main.py`
2. `app/benchmark_workbench.py` launches staged KB build or benchmark generation
3. Staged KB build uses `petroleum_kb/kb_core/pipeline/ingest.py`, `build_index.py`, and `petroleum_kb/tools/keyword_graph_from_text.py`
4. QA generation uses `petroleum_kb/eval/build_dataset.py` and `dataset_generation_utils.py`

### 6.5 Experimental Database route

1. `experimental_db_workbench.html` or `experimental_db_manager.html` sends requests to `app/main.py`
2. `app/support_layer_workbench.py` calls:
   - `sample_database()`
   - `build_petroleum_fraction_spectral_workbook_evidence()`
   - `build_existing_sample_database_evidence()`
3. `petroleum_fraction_spectral_evidence.py` loads the template definition, demo feature tables, and saved PCA/PLS models
4. Neighbor evidence is summarized and can be passed to `openai_expert_client.py`
5. Planner-routed queries use `planner_orchestrator.py` to request workbook-based historical-neighbor property inference

### 6.6 Tool onboarding route

1. `tool_onboarding.html` submits tool definitions to `app/main.py`
2. `app/tools/tool_onboarding.py` builds a capability package
3. `app/capabilities/provider_registry.py` saves, reviews, and enables it
4. The enabled provider is registered into the runtime registry
5. `generic_external_python_adapter.py` provides the released Python-script execution path

## 7. Notes

- Paths inside the public codebase are configured relative to the repository root.
- Demo assets are included for reproducibility and interface testing, but they do not represent the full private laboratory corpus or the full historical database.
- The keyword graph and QA benchmark are released as evaluation resources; the repository does not redistribute the full source-paper texts.
- Some runtime directories may be created automatically during interactive use and do not need to be precommitted.
