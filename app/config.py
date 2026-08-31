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
LLM_MODEL = _get("LLM_MODEL", "gemini-2.5-flash")
# gemini = 正式呼叫 API；mock = 不回網路，回傳範本答案（測試用）
LLM_BACKEND = _get("LLM_BACKEND", "gemini")

# --- Embedding 與向量庫 ---
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
CHROMA_DIR = Path(_get("CHROMA_DIR", str(BASE_DIR / "data" / "chroma")))
CHROMA_COLLECTION = _get("CHROMA_COLLECTION", "knowledge_base")

# --- 文件處理與檢索 ---
KNOWLEDGE_DIR = Path(_get("KNOWLEDGE_DIR", str(BASE_DIR / "data" / "knowledge")))
CHUNK_SIZE = int(_get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "50"))
TOP_K = int(_get("TOP_K", "4"))
