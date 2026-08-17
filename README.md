# explAIned-faiss

Content-based candidate search for the explAIned recommendation loop: an HTTP service on `:8001`
and the job that builds the index it serves.

Everything that imports `faiss` lives here and nowhere else — the server loads a binary format the
builder writes, so the faiss version, the index type and its parameters have to move together.
The embedding *producers* (`embeddings.py`, `build_user_embeddings.py`, Sentence-BERT + torch)
belong to `explAIned-ml`; they are coupled to this repository only through the Redis contract below.

## What it does

```
rec:article_embedding:*  ──build_index──▶  index_data/<version>/{index.faiss, ids.json}
                                                      │
rec:user_embedding:{uid} ──────────────▶  POST /search ──▶ {ids, scores}  ──▶ :5056 orchestrator
```

`IndexFlatIP(384)` over L2-normalised vectors, so inner product is cosine similarity. Exact search,
nothing to train. IVF/HNSW only becomes worth its tuning when the corpus reaches tens of thousands
of articles.

## Redis contract

| Key | Type | Written by | Read by |
|---|---|---|---|
| `rec:article_embedding:{articleId}` | STRING, 384 little-endian float32 (1536 bytes) | `embeddings.py` (explAIned-ml, not built yet) | this repo's index builder |
| `rec:user_embedding:{userId}` | STRING, same format | `build_user_embeddings.py` (not built yet) | this service, on every `/search` |

The vector format is raw little-endian float32 — no header, no JSON. `codec.py` is the only place
that encodes or decodes it. The C# orchestrator never reads these keys: `/search` takes a `user_id`,
not a vector, precisely so that 384 floats never cross the language boundary.

## HTTP

See `CONTRACT.md`. In short: `POST /search {user_id, top_k}` → `{ids, scores}`; `204` when the user
has no embedding (a normal cold-start state, not an error); `503` when no index is loaded.
`POST /reload`, `GET /health`, `GET /metrics`.

## Running

Python **3.13** (`faiss-cpu` publishes no 3.14 wheels). If the interpreter is missing:
`uv python install 3.13 && poetry env use "$(uv python find 3.13)"` — uv only supplies the
interpreter, poetry still owns the dependencies.

```bash
poetry install
export REDIS_URL='redis://:gavno@127.0.0.1:6379/0'   # same instance as the .NET services

# 1. embeddings must exist first. Until explAIned-ml lands, use the dev fixture:
poetry run python -m explained_faiss.scripts.seed_dev_embeddings --user <uid from the JWT sub>

# 2. build the artifact
poetry run python -m explained_faiss.jobs.build_index

# 3. serve it — one worker, the index is in process memory
poetry run uvicorn explained_faiss.service.app:app --host 0.0.0.0 --port 8001 --workers 1
```

The builder is a plain module with no orchestrator around it: run it by hand, from cron or from a
systemd timer. Re-running it writes a new version directory and moves `manifest.json` atomically;
the service picks the new version up within `RELOAD_WATCH_SECONDS` (or immediately on `POST /reload`).

`seed_dev_embeddings.py` is a **fixture, not ML** — deterministic pseudo-random vectors derived from
the article id. Delete it once `embeddings.py` exists.

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `REDIS_URL` | `redis://:gavno@127.0.0.1:6379/0` | same instance the .NET services use |
| `INDEX_DIR` | `index_data` | where versions and `manifest.json` live |
| `EMBEDDING_DIM` | `384` | must match the Sentence-BERT model in explAIned-ml |
| `DEFAULT_TOP_K` | `200` | ceiling for the `top_k` a caller may ask for |
| `RELOAD_WATCH_SECONDS` | `60` | background manifest check; `0` disables it |
| `KEEP_VERSIONS` | `3` | version directories the builder keeps |
| `PORT` | `8001` | |

## Tests

```bash
poetry run pytest
poetry run ruff check .
```

No Redis, no network and no prebuilt artifact needed — the API tests drive the app through
`TestClient` with a fake store and build their index in a temp directory.
