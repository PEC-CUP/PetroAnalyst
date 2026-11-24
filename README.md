# PetroAnalyst

PetroAnalyst is a traceable LLM-assisted framework for inferring and interpreting the properties of complex petroleum fractions from a historical experimental database. This release provides the source code and demo assets needed to run the workflow described in the manuscript, together with a ready-to-run demonstration system and executable.

The repository is organized as a runnable research release rather than a raw development snapshot. The capabilities described in the manuscript are retained in the public code path:

- `Historical experimental database`: template-based database construction and historical-data-based property inference for new samples
- `Domain knowledge base`: vector-indexed petroleum literature knowledge base and its keyword visualization graph
- `Literature-grounded RAG`: three routes (`naive`, `qa_oriented`, `iterative_review`) that supply document-supported interpretation of database-derived inference results
- `Pretrained feature-processing and optional tool-generated evidence`: reused GC-FID / IR PCA-PLS feature tools and the reviewed `hydrocarbon_type_calculator` provider

## Included release assets

This repository includes the minimum assets needed to inspect and test the published workflow:

- the FastAPI application and support-layer interfaces
- the Windows executable `PetroAnalyst.exe` for the reviewer workspace
- a standalone static support-layer demo page for no-install visual preview
- the released six QA datasets (1100 items in total)
- keyword-graph / keyword-visualization assets derived from the 977-paper literature corpus
- a quick-access static copy of the released topic keyword graph
- demo knowledge-base artifacts required for RAG workbench testing
- demo historical experimental database assets required for workflow execution
- embedded GC-FID and IR dimensionality-reduction models used by the historical-database workflow
- the reviewed `hydrocarbon_type_calculator` provider package and its onboarding metadata

The full private development environment, the full experimental database, and large local-only intermediate artifacts are intentionally not included.

## Main capabilities

### 1. Literature-grounded RAG

PetroAnalyst supports petroleum-domain question answering with three routes of increasing evidence organization:

- `naive`: direct retrieval and answer generation from retrieved chunks
- `qa_oriented`: query refinement, retrieval fusion, reranking, and evidence-organized answer generation
- `iterative_review`: an additional review loop that checks whether retrieved evidence is sufficient and triggers another retrieval pass when needed

These routes can be tested from the PetroAnalyst RAG workbench and can also be called through the planner workflow.

### 2. Historical experimental database

The Experimental Database workflow supports construction of a sample-centered historical database from an uploaded template, standardizes query workbooks, extracts structured numerical features, applies the embedded GC-FID / IR PCA-PLS feature tools when needed, retrieves similar historical samples, and organizes their measured properties into evidence centers and ranges for LLM-assisted property inference.

The release repository contains demo database assets so that the workflow can be executed and inspected without the full local laboratory database.

### 3. Domain knowledge base and keyword visualization

The knowledge-base workflow builds a vector-indexed petroleum literature knowledge base from a source-article corpus and generates its keyword visualization graph. The release includes the keyword-graph assets derived from the 977-paper corpus and the six-set QA benchmark (1,100 items) used for manuscript evaluation.

### 4. Support-layer interfaces and optional analytical tools

The repository includes the support-layer web interfaces used to configure and test the workflow modules:

- `/support-layer`: overview page for all modules
- `/benchmark-workbench`: KB construction and QA generation workbench
- `/rag-workbench`: PetroAnalyst RAG workbench
- `/experimental-db-workbench`: historical property inference workbench
- `/experimental-db-manager`: historical database management page
- `/tool-onboarding`: optional analytical tool onboarding and runtime test page

Pretrained GC-FID / IR feature-processing tools and the reviewed `hydrocarbon_type_calculator` provider can contribute tool-generated evidence to the same traceable workflow.

## No-install preview assets

Some released resources can be opened directly after downloading the repository, without installing dependencies or starting the application:

- `static_demo/support_layer_demo.html`: standalone visual overview of the support layer for quick demonstration only
- `static_demo/topic_keyword_graph_demo.html`: quick-access copy of the released topic keyword graph
- `petroleum_kb/eval/datasets/`: the released six QA benchmark files in `jsonl` format
- `petroleum_kb/kb_core/data/analysis/keyword_graph_text_clean/`: original keyword-graph release assets

The static support-layer demo is intentionally non-interactive. It is provided only to show the module layout and released capability scope when the runtime application is not started.

## Quick start

### Option A: Windows executable (recommended)

Double-click `PetroAnalyst.exe`. The executable starts the local workspace, opens the browser, and provides the same database, knowledge-base and RAG workbenches as the source release. Enter the reviewer API key in the in-page API settings; it is stored only in `%LOCALAPPDATA%\PetroAnalyst\.env`.

### Option B: Source mode

### 1. Create a Python environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the repository root. If `.env.example` is provided, copy it first and then fill in the values.

Typical settings include:

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_api_base_url
OPENAI_MODEL=your_model_name
```

Some workflows can be inspected without a live model call, but RAG answering, planner execution, provider review, and LLM-assisted interpretation require valid model access.

### 4. Start the application

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/support-layer` in the browser.

## Recommended evaluation flow

For first-time inspection, the following order is recommended:

1. Open `/support-layer` to inspect the four module entries.
2. Open `/rag-workbench` and run the three RAG modes on released QA examples.
3. Open `/experimental-db-workbench` and test historical-neighbor property inference with the demo workbook/template.
4. Open `/tool-onboarding` and run the hydrocarbon type calculator provider test.
5. Open `/benchmark-workbench` to inspect the QA benchmark assets and KB-related utilities.

## Optional knowledge-base dependencies

The root `requirements.txt` already contains the dependencies needed for the standard public workflow. Some optional KB rebuild utilities may additionally require:

- `spacy`: only needed when rebuilding the KB with sentence-based chunking modes such as `scientific_sentence`
- `requests`: only needed for helper scripts such as `petroleum_kb/check_grobid.py`

If the goal is only to run the released demo workflows and interfaces, no extra dependency file is required.

## Project structure

A detailed release map is provided in [RELEASE_STRUCTURE.md](RELEASE_STRUCTURE.md).

High-level layout:

```text
PetroAnalyst/
- app/                FastAPI app, planner, RAG, tools, and static support-layer pages
- data/               Demo Experimental Database assets and related runtime resources
- petroleum_kb/       KB processing package, retrieval core, demo index assets, and QA datasets
- static_demo/        No-install preview pages and quick-access keyword graph copy
- .env.example        Example environment configuration
- README.md
- RELEASE_STRUCTURE.md
- requirements.txt
```

## Notes for public users

- Paths inside the public codebase are configured relative to the repository root.
- Demo assets are included for reproducibility and interface testing, but they do not represent the full private laboratory corpus or the full historical database.
- The keyword graph and QA benchmark are released as evaluation resources; the repository does not redistribute the full source-paper texts.
- The support layer is provided as a convenient modular interface for use, testing, and secondary development of similar analytical workflows.

## Citation and release scope

If this repository is used in research, please cite the associated manuscript after publication.

The public repository is intended to reproduce the manuscript-aligned PetroAnalyst workflow, inspect the support-layer interfaces, and provide a reusable starting point for related LLM-assisted analytical systems.

If complete-function scripts or more detailed technical discussion are needed, please contact `sq@cup.edu.cn` or `lsf@cup.edu.cn`.
