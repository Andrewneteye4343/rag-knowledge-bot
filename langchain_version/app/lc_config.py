"""設定檔：與主專案共用 .env（位於專案根目錄）。"""
import os
from pathlib import Path

from dotenv import load_dotenv

# langchain_version/ 的上一層 = 專案根目錄
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR.parent
load_dotenv(PROJECT_DIR / ".env")


def _get(key: str, default: str) -> str:
    return os.getenv(key, default)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = _get("LLM_MODEL", "gemini-3.6-flash")

EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
CACHE_DIR = Path(_get("CACHE_DIR", str(PROJECT_DIR / "cache")))        # 與主專案共用模型快取
KNOWLEDGE_DIR = Path(_get("KNOWLEDGE_DIR", str(PROJECT_DIR / "data" / "knowledge")))
CHROMA_DIR = Path(_get("CHROMA_DIR", str(PROJECT_DIR / "data" / "chroma_lc")))  # 獨立索引，避免與主專案衝突

CHUNK_SIZE = int(_get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "50"))
TOP_K = int(_get("TOP_K", "4"))

# --- Reranker 二階段檢索（與主專案對等）---
RERANK_TOP_K = int(_get("RERANK_TOP_K", "20"))  # 海選數量；<= TOP_K 即停用
RERANK_MODEL = _get("RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
