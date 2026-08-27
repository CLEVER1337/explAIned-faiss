# faiss-cpu ships wheels for cp312/cp313 only — do not bump this to 3.14 until it does.
FROM python:3.12-slim

# POETRY_NO_CACHE as well as PIP_NO_CACHE_DIR: two separate caches, and disabling pip's does
# nothing about /root/.cache/pypoetry, where poetry keeps every wheel it downloaded. Those
# stayed in the layer beside the unpacked copy — faiss-cpu and numpy twice over.
#
# Found in explAIned-ml, where the same two lines made the jobs image large enough to fill
# the CI runner's disk mid-export. Nothing failed here; this image is simply smaller. Kept
# identical to the ml one on purpose — the two Dockerfiles are already near-copies, and a
# fix that lands in one and not the other is how they stop being.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_CACHE=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN pip install "poetry>=1.8"

COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-root \
    && rm -rf /root/.cache

# README.md is not documentation here: pyproject declares `readme`, and poetry-core refuses to
# build the root package if the file is missing.
COPY README.md ./
COPY src ./src
RUN poetry install --only-root \
    && rm -rf /root/.cache

ENV INDEX_DIR=/var/lib/explained-faiss
VOLUME ["/var/lib/explained-faiss"]

EXPOSE 8001

# One worker on purpose: the index lives in process memory, every worker is another full copy.
CMD ["uvicorn", "explained_faiss.service.app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
