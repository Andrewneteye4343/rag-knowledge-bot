"""文件載入：直接重用主專案 loader.py（同一個解析邏輯），
並把結果轉成 LangChain 的 Document 物件。

教學重點：框架元件與自訂/既有程式碼可以混用——不是什麼都要用框架的。
"""
import sys
from pathlib import Path

from langchain_core.documents import Document

# 讓 loader.py（主專案）可以被 import
APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from loader import load_all as _load_all_texts  # noqa: E402


def load_documents(knowledge_dir: Path) -> list[Document]:
    """讀取知識庫文件並回傳 Document 列表（保留來源檔名作為 metadata）。"""
    docs = _load_all_texts(knowledge_dir)
    return [
        Document(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in docs
    ]
