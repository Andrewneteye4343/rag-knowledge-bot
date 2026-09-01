FROM python:3.11-slim

# 避免 pycache 與 pip 快取佔空間
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 依賴（先裝系統工具再裝 Python 套件）
# tesseract-ocr：掃描型 PDF 的 OCR 辨識；poppler-utils：pdf2image 轉圖用
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-chi-tra \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴清單並安裝（善用 Docker 層快取）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式與資料
COPY app/ ./app/
COPY data/ ./data/
RUN mkdir -p /app/data/chroma

WORKDIR /app/app

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
