FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REGDOC_PROJECT_ROOT=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-core.txt requirements-app.txt requirements-infra.txt pyproject.toml README.md ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements-infra.txt
COPY . .
RUN python -m pip install -e .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
