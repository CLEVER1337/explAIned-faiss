"""The index artifact manifest.

`INDEX_DIR` looks like:

    index_data/
      manifest.json          <- written last, atomically; the switchover point
      20260816T184500Z/
        index.faiss
        ids.json

Readers never scan for versions: they read `manifest.json` and load the directory it names.
A half-written version directory is therefore invisible until the manifest moves.
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class Manifest:
    version: str
    built_at: str
    dim: int
    count: int
    metric: str
    faiss_version: str

    @property
    def built_at_utc(self) -> datetime:
        return datetime.fromisoformat(self.built_at)

    @property
    def age_seconds(self) -> float:
        return (datetime.now(UTC) - self.built_at_utc).total_seconds()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Manifest":
        return cls(
            version=payload["version"],
            built_at=payload["built_at"],
            dim=int(payload["dim"]),
            count=int(payload["count"]),
            metric=payload.get("metric", "ip"),
            faiss_version=payload.get("faiss_version", "unknown"),
        )


def new_version() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def manifest_path(index_dir: Path) -> Path:
    return index_dir / MANIFEST_NAME


def read_manifest(index_dir: Path) -> Manifest | None:
    """Returns None when there is no manifest, or when it is unreadable/malformed.

    A broken manifest must not take the service down: it keeps serving whatever it has.
    """
    path = manifest_path(index_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None

    try:
        return Manifest.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


def write_manifest(index_dir: Path, manifest: Manifest) -> Path:
    """Atomic replace: readers see either the old manifest or the new one, never a partial file."""
    index_dir.mkdir(parents=True, exist_ok=True)
    target = manifest_path(index_dir)

    fd, tmp_name = tempfile.mkstemp(dir=index_dir, prefix=".manifest-", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    return target
