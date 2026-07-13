"""PickleModelRepository — file-backed ModelRepository using pickle.

Persists a whole trained Detector object. Pickle is fine for a POC / trusted
local artifacts; a production deployment would swap this adapter for a versioned,
non-executable serialization without touching the use case (it depends on the port).
"""

from __future__ import annotations

import pickle
from pathlib import Path

from waf.domain.port.detector import Detector


class PickleModelRepository:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def save(self, detector: Detector) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(pickle.dumps(detector))

    def load(self) -> Detector:
        if not self._path.exists():
            raise FileNotFoundError(f"no model saved at {self._path}")
        detector: Detector = pickle.loads(self._path.read_bytes())
        return detector
