FROM python:3.11-slim

# 避免 pycache 與 pip 快取佔空間
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 先裝依賴（善用 Docker 層快取）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式與資料
COPY app/ ./app/
COPY data/ ./data/
RUN mkdir -p /app/data/chroma

WORKDIR /app/app

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
