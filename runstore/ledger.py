"""Append-only hash-chained audit ledger.

Every artifact mutation is recorded as a hash-chained entry. Tampering with
any entry breaks the chain at the next step, providing verifiable provenance
for insurance claim packs and regulatory audit.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from domain import AuditEntry

LEDGER_DB = Path(__file__).resolve().parent.parent / "data" / "audit_ledger.db"


def _hash_entry(entry_dict: dict) -> str:
    """Deterministic SHA-256 of a ledger entry's content fields."""
    canonical = json.dumps(entry_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditLedger:
    """Append-only hash-chained audit ledger backed by SQLite.

    Schema per entry:
        prev_hash, ts, actor, kind, artifact_hash,
        prompt_version, model_version, sim_params, seed

    The ``entry_hash`` column stores the SHA-256 of the full entry (including
    ``prev_hash``), creating a tamper-evident chain.
    """

    GENESIS_HASH = "0" * 64  # genesis block prev_hash

    def __init__(self, db_path: Path = LEDGER_DB):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ledger (
                    seq           INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_hash    TEXT NOT NULL UNIQUE,
                    prev_hash     TEXT NOT NULL,
                    ts            TEXT NOT NULL,
                    actor         TEXT NOT NULL,
                    kind          TEXT NOT NULL,
                    artifact_hash TEXT NOT NULL,
                    prompt_version TEXT DEFAULT '',
                    model_version  TEXT DEFAULT '',
                    sim_params    TEXT DEFAULT '{}',
                    seed          INTEGER DEFAULT 0
                )
            """)

    def _last_hash(self) -> str:
        """Return the hash of the most recent ledger entry, or genesis hash."""
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT entry_hash FROM ledger ORDER BY seq DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else self.GENESIS_HASH

    def append(
        self,
        actor: str,
        kind: str,
        artifact_hash: str,
        prompt_version: str = "",
        model_version: str = "",
        sim_params: dict | None = None,
        seed: int = 0,
    ) -> AuditEntry:
        """Append a new entry to the ledger, chaining it to the previous hash."""
        prev_hash = self._last_hash()
        now = datetime.now(timezone.utc)
        sim_params = sim_params or {}

        entry = AuditEntry(
            prev_hash=prev_hash,
            ts=now,
            actor=actor,
            kind=kind,
            artifact_hash=artifact_hash,
            prompt_version=prompt_version,
            model_version=model_version,
            sim_params=sim_params,
            seed=seed,
        )

        entry_dict = asdict(entry)
        entry_hash = _hash_entry(entry_dict)

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO ledger "
                "(entry_hash, prev_hash, ts, actor, kind, artifact_hash, "
                "prompt_version, model_version, sim_params, seed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry_hash,
                    prev_hash,
                    now.isoformat(),
                    actor,
                    kind,
                    artifact_hash,
                    prompt_version,
                    model_version,
                    json.dumps(sim_params, default=str),
                    seed,
                ),
            )

        return entry

    def chain(self) -> list[dict]:
        """Return the full ledger chain as a list of dicts (oldest first)."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM ledger ORDER BY seq ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def verify_chain(self) -> tuple[bool, int]:
        """Verify the entire hash chain.

        Returns (is_valid, first_broken_seq). If valid, first_broken_seq is -1.
        """
        entries = self.chain()
        if not entries:
            return True, -1

        expected_prev = self.GENESIS_HASH
        for entry in entries:
            if entry["prev_hash"] != expected_prev:
                return False, entry["seq"]

            # Reconstruct the AuditEntry dict and re-hash
            reconstructed = {
                "prev_hash": entry["prev_hash"],
                "ts": entry["ts"],
                "actor": entry["actor"],
                "kind": entry["kind"],
                "artifact_hash": entry["artifact_hash"],
                "prompt_version": entry["prompt_version"],
                "model_version": entry["model_version"],
                "sim_params": json.loads(entry["sim_params"]) if isinstance(entry["sim_params"], str) else entry["sim_params"],
                "seed": entry["seed"],
            }
            computed_hash = _hash_entry(reconstructed)
            if computed_hash != entry["entry_hash"]:
                return False, entry["seq"]

            expected_prev = entry["entry_hash"]

        return True, -1

    def tamper(self, seq: int, field: str, new_value: Any) -> None:
        """Tamper with a ledger entry for demo / testing purposes only."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                f"UPDATE ledger SET {field} = ? WHERE seq = ?",
                (str(new_value), seq),
            )

    def get_entries_for_artifact(self, artifact_hash: str) -> list[dict]:
        """Get all ledger entries referencing a specific artifact hash."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM ledger WHERE artifact_hash = ? ORDER BY seq ASC",
                (artifact_hash,),
            ).fetchall()
        return [dict(r) for r in rows]

    def digest(self) -> str:
        """Return the hash of the latest entry — the 'tip' of the chain."""
        return self._last_hash()
