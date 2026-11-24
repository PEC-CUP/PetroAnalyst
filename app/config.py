import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = Path(os.environ.get("PETROANALYST_ENV_FILE", str(BASE_DIR / ".env")))
load_dotenv(ENV_FILE)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "PetroAnalyst"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    deepseek_trust_env: bool = Field(default=False, alias="DEEPSEEK_TRUST_ENV")
    deepseek_chat_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_CHAT_MODEL")
    deepseek_reasoner_model: str = Field(default="deepseek-reasoner", alias="DEEPSEEK_REASONER_MODEL")
    deepseek_chat_model_fallbacks_raw: str = Field(default="", alias="DEEPSEEK_CHAT_MODEL_FALLBACKS")
    deepseek_reasoner_model_fallbacks_raw: str = Field(default="deepseek-chat", alias="DEEPSEEK_REASONER_MODEL_FALLBACKS")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-5.4-mini", alias="OPENAI_MODEL")
    openai_strong_model: str = Field(default="gpt-5.5", alias="OPENAI_STRONG_MODEL")
    openai_balanced_model: str = Field(default="gpt-5.4", alias="OPENAI_BALANCED_MODEL")
    openai_fast_model: str = Field(default="gpt-5.4-mini", alias="OPENAI_FAST_MODEL")
    openai_cheap_model: str = Field(default="gpt-5.4-nano", alias="OPENAI_CHEAP_MODEL")
    llm_timeout_seconds: float = Field(default=60.0, alias="LLM_TIMEOUT_SECONDS")
    tesseract_cmd: str = Field(default="", alias="TESSERACT_CMD")
    tessdata_prefix: str = Field(default="", alias="TESSDATA_PREFIX")

    knowledge_dir: Path = BASE_DIR / "knowledge"
    knowledge_extra_dirs_raw: str = Field(default="", alias="KNOWLEDGE_EXTRA_DIRS")
    petroleum_kb_enabled: bool = Field(default=True, alias="PETROLEUM_KB_ENABLED")
    petroleum_kb_dir: Path = BASE_DIR / "petroleum_kb"
    petroleum_kb_embedding_local_files_only: bool = Field(
        default=False,
        alias="PETROLEUM_KB_EMBEDDING_LOCAL_FILES_ONLY",
        description="Force petroleum_kb embedder to load models from local cache only (no HF network).",
    )
    petroleum_kb_enable_reranker: bool = Field(
        default=True,
        alias="PETROLEUM_KB_ENABLE_RERANKER",
        description="Enable petroleum_kb cross-encoder reranker.",
    )
    petroleum_kb_rerank_model: str = Field(
        default="",
        alias="PETROLEUM_KB_RERANK_MODEL",
        description="Override reranker model name for petroleum_kb.",
    )
    petroleum_kb_rerank_local_files_only: bool = Field(
        default=False,
        alias="PETROLEUM_KB_RERANK_LOCAL_FILES_ONLY",
        description="Force petroleum_kb reranker to load models from local cache only (no HF network).",
    )
    rag_top_k: int = 4
    chunk_max_chars: int = 800
    advanced_rag_enabled: bool = Field(default=False, alias="ADVANCED_RAG_ENABLED")
    advanced_rag_max_rounds: int = Field(default=2, alias="ADVANCED_RAG_MAX_ROUNDS")
    advanced_rag_top_docs: int = Field(default=5, alias="ADVANCED_RAG_TOP_DOCS")
    advanced_rag_top_k_chunks: int = Field(default=12, alias="ADVANCED_RAG_TOP_K_CHUNKS")
    qa_oriented_initial_top_docs: int = Field(default=12, alias="QA_ORIENTED_INITIAL_TOP_DOCS")
    qa_oriented_initial_top_k_chunks: int = Field(default=30, alias="QA_ORIENTED_INITIAL_TOP_K_CHUNKS")
    qa_oriented_chunks_per_doc: int = Field(default=4, alias="QA_ORIENTED_CHUNKS_PER_DOC")
    evidence_filter_model: str = Field(default="", alias="EVIDENCE_FILTER_MODEL")
    evidence_filter_thinking: bool = Field(default=False, alias="EVIDENCE_FILTER_THINKING")
    evidence_filter_batch_size: int = Field(default=6, alias="EVIDENCE_FILTER_BATCH_SIZE")
    evidence_filter_min_score: float = Field(default=6.0, alias="EVIDENCE_FILTER_MIN_SCORE")
    evidence_filter_top_k: int = Field(default=10, alias="EVIDENCE_FILTER_TOP_K")
    evidence_filter_max_chars: int = Field(default=1800, alias="EVIDENCE_FILTER_MAX_CHARS")

    def knowledge_source_dirs(self) -> list[Path]:
        raw_items = [item.strip() for item in self.knowledge_extra_dirs_raw.split(";") if item.strip()]
        dirs = [self.knowledge_dir]
        seen: set[str] = set()

        for item in raw_items:
            path = Path(item).expanduser()
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            dirs.append(path)

        return dirs

    def deepseek_chat_model_fallbacks(self) -> list[str]:
        return [item.strip() for item in self.deepseek_chat_model_fallbacks_raw.split(";") if item.strip()]

    def deepseek_reasoner_model_fallbacks(self) -> list[str]:
        return [item.strip() for item in self.deepseek_reasoner_model_fallbacks_raw.split(";") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    return settings
