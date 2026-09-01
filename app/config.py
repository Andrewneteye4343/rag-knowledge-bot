"""設定檔：從 .env 讀取所有可調參數。"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _get(key: str, default: str) -> str:
    return os.getenv(key, default)


# --- LLM ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = _get("LLM_MODEL", "gemini-3.6-flash")
# gemini = 正式呼叫 API；mock = 不回網路，回傳範本答案（測試用）
LLM_BACKEND = _get("LLM_BACKEND", "gemini")

# --- Embedding 與向量庫 ---
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
# 模型快取目錄（容器內為 /app/cache，對應本機 ./cache，避免每次重新下載）
CACHE_DIR = Path(_get("CACHE_DIR", str(BASE_DIR / "cache")))
CHROMA_DIR = Path(_get("CHROMA_DIR", str(BASE_DIR / "data" / "chroma")))
CHROMA_COLLECTION = _get("CHROMA_COLLECTION", "knowledge_base")

# --- 文件處理與檢索 ---
KNOWLEDGE_DIR = Path(_get("KNOWLEDGE_DIR", str(BASE_DIR / "data" / "knowledge")))
CHUNK_SIZE = int(_get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "50"))
TOP_K = int(_get("TOP_K", "4"))

# --- Reranker（二階段檢索）---
# 第一階段「海選」的候選數量；設為 <= TOP_K 即停用 rerank
RERANK_TOP_K = int(_get("RERANK_TOP_K", "20"))
# 多語言 reranker（中英皆可，第一次使用下載約 1.1GB）
RERANK_MODEL = _get("RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")

# --- OCR（掃描型 PDF 用）---
OCR_LANG = _get("OCR_LANG", "chi_tra+eng")
