"""Content-addressed artifact store with SHA-256 hashing.

Artifacts are stored by their content hash (SHA-256 of the serialized bytes),
making every artifact immutable and replay-deterministic. A SQLite index tracks
metadata for queryable run provenance.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARTIFACT_ROOT = Path(__file__).resolve().parent.parent / "data" / "runs"
INDEX_DB = Path(__file__).resolve().parent.parent / "data" / "runstore_index.db"


def _content_hash(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ContentRef:
    """Reference to a content-addressed artifact."""
    run_id: str
    kind: str
    key: str
    content_hash: str
    created_at: str


class ContentAddressedStore:
    """Filesystem-backed content-addressed store with SQLite index.

    Each artifact is stored as ``<root>/<run_id>/<kind>/<content_hash>.pkl``
    and indexed in a SQLite database for fast lookup.
    """

    def __init__(self, root: Path = ARTIFACT_ROOT, index_db: Path = INDEX_DB):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_db = index_db
        self.index_db.parent.mkdir(parents=True, exist_ok=True)
        self._init_index()

    def _init_index(self) -> None:
        with sqlite3.connect(str(self.index_db)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id      TEXT NOT NULL,
                    kind        TEXT NOT NULL,
                    key         TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    PRIMARY KEY (run_id, kind, key)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_artifacts_run
                ON artifacts(run_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_artifacts_hash
                ON artifacts(content_hash)
            """)

    def _dir(self, run_id: str, kind: str) -> Path:
        d = self.root / run_id / kind
        d.mkdir(parents=True, exist_ok=True)
        return d

    def put(self, run_id: str, kind: str, artifact: Any, key: str | None = None) -> ContentRef:
        """Store an artifact and return a content-addressed reference."""
        data = pickle.dumps(artifact)
        h = _content_hash(data)
        key = key or f"{kind}_{h[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        # Write to filesystem (content-addressed — idempotent)
        blob_path = self._dir(run_id, kind) / f"{h}.pkl"
        if not blob_path.exists():
            blob_path.write_bytes(data)

        # Also write a symlink/alias by key for convenient access
        key_path = self._dir(run_id, kind) / f"{key}.pkl"
        if not key_path.exists():
            key_path.write_bytes(data)

        # Index
        with sqlite3.connect(str(self.index_db)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO artifacts (run_id, kind, key, content_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, kind, key, h, now),
            )

        return ContentRef(run_id=run_id, kind=kind, key=key, content_hash=h, created_at=now)

    def get(self, run_id: str, kind: str, key: str) -> Any:
        """Retrieve an artifact by run_id/kind/key."""
        path = self._dir(run_id, kind) / f"{key}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {run_id}/{kind}/{key}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_by_hash(self, run_id: str, kind: str, content_hash: str) -> Any:
        """Retrieve an artifact by its content hash."""
        path = self._dir(run_id, kind) / f"{content_hash}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {content_hash}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def verify(self, run_id: str, kind: str, key: str) -> bool:
        """Verify integrity: recompute hash and compare with index."""
        path = self._dir(run_id, kind) / f"{key}.pkl"
        if not path.exists():
            return False
        data = path.read_bytes()
        actual_hash = _content_hash(data)
        with sqlite3.connect(str(self.index_db)) as conn:
            row = conn.execute(
                "SELECT content_hash FROM artifacts WHERE run_id=? AND kind=? AND key=?",
                (run_id, kind, key),
            ).fetchone()
        if not row:
            return False
        return actual_hash == row[0]

    def list_artifacts(self, run_id: str, kind: str | None = None) -> list[ContentRef]:
        """List all artifacts for a run, optionally filtered by kind."""
        with sqlite3.connect(str(self.index_db)) as conn:
            if kind:
                rows = conn.execute(
                    "SELECT run_id, kind, key, content_hash, created_at "
                    "FROM artifacts WHERE run_id=? AND kind=? ORDER BY created_at",
                    (run_id, kind),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT run_id, kind, key, content_hash, created_at "
                    "FROM artifacts WHERE run_id=? ORDER BY created_at",
                    (run_id,),
                ).fetchall()
        return [ContentRef(*r) for r in rows]

    def list_runs(self) -> list[str]:
        """List all run IDs in the index."""
        with sqlite3.connect(str(self.index_db)) as conn:
            rows = conn.execute(
                "SELECT DISTINCT run_id FROM artifacts ORDER BY run_id"
            ).fetchall()
        return [r[0] for r in rows]
