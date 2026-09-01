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
CACHE_DIR = PROJECT_DIR / "cache"          # 與主專案共用模型快取
KNOWLEDGE_DIR = PROJECT_DIR / "data" / "knowledge"
CHROMA_DIR = PROJECT_DIR / "data" / "chroma_lc"   # 獨立索引，避免與主專案衝突

CHUNK_SIZE = int(_get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "50"))
TOP_K = int(_get("TOP_K", "4"))
