# LangChain 版 Dockerfile（build context 為專案根目錄）
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# 系統工具（與主專案相同：OCR 用）
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-chi-tra \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# LangChain 依賴
COPY langchain_version/requirements.txt /tmp/lc-requirements.txt
RUN pip install --no-cache-dir -r /tmp/lc-requirements.txt

# 共用主專案的 app/（loader.py 等）與 LangChain 版程式
COPY app/ ./app/
COPY langchain_version/app/ ./langchain_version/app/

WORKDIR /app/langchain_version/app

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
