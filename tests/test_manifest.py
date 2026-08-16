import json

from explained_faiss.manifest import MANIFEST_NAME, Manifest, manifest_path, read_manifest, write_manifest


def a_manifest(**overrides) -> Manifest:
    base = {
        "version": "20260816T120000Z",
        "built_at": "2026-08-16T12:00:00+00:00",
        "dim": 384,
        "count": 12,
        "metric": "ip",
        "faiss_version": "1.9.0",
    }
    return Manifest(**{**base, **overrides})


def test_write_then_read(tmp_path):
    written = a_manifest()

    write_manifest(tmp_path, written)

    assert read_manifest(tmp_path) == written


def test_missing_manifest_reads_as_none(tmp_path):
    assert read_manifest(tmp_path) is None


def test_malformed_manifest_reads_as_none(tmp_path):
    manifest_path(tmp_path).write_text("{not json", encoding="utf-8")

    assert read_manifest(tmp_path) is None


def test_manifest_without_required_field_reads_as_none(tmp_path):
    manifest_path(tmp_path).write_text(json.dumps({"version": "v1"}), encoding="utf-8")

    assert read_manifest(tmp_path) is None


def test_replacement_is_atomic(tmp_path):
    write_manifest(tmp_path, a_manifest(version="old"))
    write_manifest(tmp_path, a_manifest(version="new", count=99))

    # No temp files left behind, and readers only ever see a complete manifest.
    assert [path.name for path in tmp_path.iterdir()] == [MANIFEST_NAME]
    assert read_manifest(tmp_path).version == "new"


def test_age_is_measured_from_built_at():
    from datetime import UTC, datetime, timedelta

    built = datetime.now(UTC) - timedelta(seconds=90)

    assert 89 <= a_manifest(built_at=built.isoformat()).age_seconds <= 120
