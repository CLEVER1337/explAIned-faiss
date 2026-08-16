# faiss-cpu ships wheels for cp312/cp313 only — do not bump this to 3.14 until it does.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN pip install "poetry>=1.8"

COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-root

COPY src ./src
RUN poetry install --only-root

ENV INDEX_DIR=/var/lib/explained-faiss
VOLUME ["/var/lib/explained-faiss"]

EXPOSE 8001

# One worker on purpose: the index lives in process memory, every worker is another full copy.
CMD ["uvicorn", "explained_faiss.service.app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
