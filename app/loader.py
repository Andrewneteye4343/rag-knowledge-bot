"""文件載入與切塊（chunking）。

支援 .txt / .md / .pdf / .docx。
切塊策略：段落 -> 句子 -> 硬切，並保留前後重疊，避免語意被切斷。
"""
import re
from pathlib import Path

import pypdf
from docx import Document

SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx"}

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])")


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf(path: Path) -> str:
    reader = pypdf.PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())


_LOADERS = {
    ".txt": load_text_file,
    ".md": load_text_file,
    ".pdf": load_pdf,
    ".docx": load_docx,
}


def load_all(knowledge_dir: Path) -> list[dict]:
    """讀取目錄下所有支援的文件，回傳 [{source, text}]。"""
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"知識庫目錄不存在: {knowledge_dir}")

    docs = []
    for path in sorted(knowledge_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        text = _LOADERS[path.suffix.lower()](path).strip()
        if text:
            docs.append({"source": path.name, "text": text})
            print(f"  ✓ {path.name}（{len(text)} 字元）")
        else:
            print(f"  ⚠ {path.name}（無法讀取內容，已跳過）")
    return docs


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """遞迴式切塊：先依空白行切段落，再依句子切，最後硬切。"""
    chunks: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    def _hard_cut(segment: str) -> None:
        """超過 chunk_size 的片段：依句子切，仍過長就硬切。"""
        seg = segment.strip()
        if not seg:
            return
        if len(seg) <= chunk_size:
            chunks.append(seg)
            return
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(seg) if s.strip()]
        buf = ""
        for s in sentences:
            if len(s) > chunk_size:  # 單句就過長 -> 硬切
                if buf:
                    chunks.append(buf)
                    buf = ""
                while s:
                    chunks.append(s[:chunk_size])
                    s = s[chunk_size:]
            elif len(buf) + len(s) + 1 <= chunk_size:
                buf += s
            else:
                chunks.append(buf)
                buf = s
        if buf:
            chunks.append(buf)

    buf = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            if buf:
                chunks.append(buf)
                buf = ""
            _hard_cut(para)
        elif len(buf) + len(para) + 1 <= chunk_size:
            buf += para + "\n"
        else:
            chunks.append(buf)
            buf = para + "\n"
    if buf:
        chunks.append(buf)

    # 加入重疊：下一個 chunk 開頭補上前一個 chunk 的結尾
    result = []
    for i, chunk in enumerate(chunks):
        if i > 0 and overlap > 0:
            chunk = chunks[i - 1][-overlap:] + chunk
        result.append(chunk.strip())
    return result


def make_chunks(docs: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """將文件們切成區塊，回傳 [{id, source, chunk_index, text}]。"""
    chunks: list[dict] = []
    for doc in docs:
        source_id = re.sub(r"[^\w\-.]", "_", doc["source"])
        for idx, text in enumerate(split_text(doc["text"], chunk_size, overlap)):
            chunks.append({
                "id": f"{source_id}__{idx}",
                "source": doc["source"],
                "chunk_index": idx,
                "text": text,
            })
    return chunks
