"""Accumulate trunk detections across frames and build a persistent trunk map."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import KDTree

from tree_trunk_mapper.detector import TrunkDetection


@dataclass
class TrunkRecord:
    """A single trunk in the persistent map."""

    trunk_id: int
    position: np.ndarray  # (3,) mean position
    dbh: float  # mean diameter at breast height
    observation_count: int = 1
    species: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "trunk_id": self.trunk_id,
            "position": self.position.tolist(),
            "dbh": float(self.dbh),
            "observation_count": self.observation_count,
            "species": self.species,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrunkRecord:
        return cls(
            trunk_id=d["trunk_id"],
            position=np.array(d["position"]),
            dbh=d["dbh"],
            observation_count=d.get("observation_count", 1),
            species=d.get("species", "unknown"),
        )


class TrunkMapper:
    """Accumulate detections and merge nearby ones into a persistent trunk map.

    Parameters
    ----------
    merge_radius : float
        Detections within this distance (metres) of an existing trunk are merged
        rather than creating a new entry.
    """

    def __init__(self, merge_radius: float = 0.5) -> None:
        self.merge_radius = merge_radius
        self._trunks: list[TrunkRecord] = []
        self._next_id: int = 0
        # Running sums for incremental mean computation
        self._position_sums: list[np.ndarray] = []
        self._dbh_sums: list[float] = []

    def add_detections(self, detections: list[TrunkDetection]) -> None:
        """Integrate a list of detections from a single frame into the map."""
        if not detections:
            return

        for det in detections:
            self._integrate_detection(det)

    def _integrate_detection(self, det: TrunkDetection) -> None:
        if not self._trunks:
            self._create_trunk(det)
            return

        # Find nearest existing trunk
        positions = np.array([t.position for t in self._trunks])
        tree = KDTree(positions[:, :2])  # match in xy only
        dist, idx = tree.query(det.center[:2])

        if dist <= self.merge_radius:
            self._merge_into(idx, det)
        else:
            self._create_trunk(det)

    def _create_trunk(self, det: TrunkDetection) -> None:
        trunk = TrunkRecord(
            trunk_id=self._next_id,
            position=det.center.copy(),
            dbh=det.dbh,
            observation_count=1,
        )
        self._trunks.append(trunk)
        self._position_sums.append(det.center.copy())
        self._dbh_sums.append(det.dbh)
        self._next_id += 1

    def _merge_into(self, idx: int, det: TrunkDetection) -> None:
        trunk = self._trunks[idx]
        trunk.observation_count += 1

        # Incremental mean update
        self._position_sums[idx] += det.center
        self._dbh_sums[idx] += det.dbh

        trunk.position = self._position_sums[idx] / trunk.observation_count
        trunk.dbh = self._dbh_sums[idx] / trunk.observation_count

    def get_map(self) -> list[TrunkRecord]:
        """Return the current trunk map as a list of TrunkRecord."""
        return list(self._trunks)
