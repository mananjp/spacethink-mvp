"""Filesystem-backed artifact store keyed by run_id. Simple JSON/pickle backing.

This is the integration seam described in the parallel-work-split doc: every
stage reads/writes typed artifacts through RunStore instead of passing objects
directly, so stages can be swapped or run independently.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARTIFACT_ROOT = Path(__file__).resolve().parent.parent / "data" / "runs"


@dataclass(frozen=True)
class ArtifactRef:
    run_id: str
    kind: str
    key: str


class RunStore:
    def __init__(self, root: Path = ARTIFACT_ROOT):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, run_id: str, kind: str) -> Path:
        d = self.root / run_id / kind
        d.mkdir(parents=True, exist_ok=True)
        return d

    def put(self, run_id: str, kind: str, artifact: Any, key: str | None = None) -> ArtifactRef:
        key = key or f"{kind}_{len(list(self._dir(run_id, kind).iterdir()))}"
        path = self._dir(run_id, kind) / f"{key}.pkl"
        with open(path, "wb") as f:
            pickle.dump(artifact, f)
        return ArtifactRef(run_id=run_id, kind=kind, key=key)

    def get(self, ref: ArtifactRef) -> Any:
        path = self._dir(ref.run_id, ref.kind) / f"{ref.key}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    def list(self, run_id: str, kind: str) -> list[ArtifactRef]:
        d = self._dir(run_id, kind)
        return [ArtifactRef(run_id, kind, p.stem) for p in sorted(d.glob("*.pkl"))]
