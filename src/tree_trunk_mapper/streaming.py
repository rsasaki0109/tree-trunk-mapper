"""Real-time streaming detection: process LiDAR frames as they arrive."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import open3d as o3d

from tree_trunk_mapper.detector import TrunkDetection, detect_trunks
from tree_trunk_mapper.mapper import TrunkMapper, TrunkRecord


@dataclass
class StreamingResult:
    """Result of processing a single streaming frame."""

    new_detections: list[TrunkDetection]
    merged_count: int
    total_trunks: int
    trunk_map: list[TrunkRecord]
    frame_count: int
    processing_time_ms: float


class StreamingDetector:
    """Wraps TrunkMapper + detect_trunks for incremental real-time processing.

    Parameters
    ----------
    merge_radius : float
        Radius for merging nearby detections across frames.
    process_every_n : int
        Process every N-th frame (skip the rest for performance). Default 1 = no skip.
    on_detection : callable or None
        Optional callback invoked with each ``TrunkDetection`` as it is found.
    **detect_kwargs
        Additional keyword arguments forwarded to ``detect_trunks``.
    """

    def __init__(
        self,
        merge_radius: float = 0.5,
        process_every_n: int = 1,
        on_detection: Callable[[TrunkDetection], None] | None = None,
        **detect_kwargs: object,
    ) -> None:
        if process_every_n < 1:
            raise ValueError("process_every_n must be >= 1")

        self._mapper = TrunkMapper(merge_radius=merge_radius)
        self._process_every_n = process_every_n
        self._on_detection = on_detection
        self._detect_kwargs = detect_kwargs

        # Stats
        self._total_frames: int = 0
        self._processed_frames: int = 0
        self._processing_times: list[float] = []

    # --- public API ---

    def process_frame(self, points: np.ndarray) -> StreamingResult:
        """Detect trunks in a frame and accumulate into the map.

        Parameters
        ----------
        points : np.ndarray
            (N, 3+) point array for one LiDAR frame.

        Returns
        -------
        StreamingResult
        """
        self._total_frames += 1

        # Frame skipping
        if self._total_frames % self._process_every_n != 0:
            trunk_map = self._mapper.get_map()
            return StreamingResult(
                new_detections=[],
                merged_count=0,
                total_trunks=len(trunk_map),
                trunk_map=trunk_map,
                frame_count=self._total_frames,
                processing_time_ms=0.0,
            )

        t0 = time.perf_counter()

        # Build Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))

        # Detect
        detections = detect_trunks(pcd, **self._detect_kwargs)  # type: ignore[arg-type]

        # Callback
        if self._on_detection is not None:
            for det in detections:
                self._on_detection(det)

        # Record map size before merge
        prev_count = len(self._mapper.get_map())

        # Accumulate
        self._mapper.add_detections(detections)

        new_count = len(self._mapper.get_map())
        merged_count = len(detections) - (new_count - prev_count)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._processed_frames += 1
        self._processing_times.append(elapsed_ms)

        trunk_map = self._mapper.get_map()
        return StreamingResult(
            new_detections=detections,
            merged_count=merged_count,
            total_trunks=len(trunk_map),
            trunk_map=trunk_map,
            frame_count=self._total_frames,
            processing_time_ms=elapsed_ms,
        )

    def get_map(self) -> list[TrunkRecord]:
        """Return the current accumulated trunk map."""
        return self._mapper.get_map()

    def reset(self) -> None:
        """Clear the map and all statistics."""
        self._mapper = TrunkMapper(merge_radius=self._mapper.merge_radius)
        self._total_frames = 0
        self._processed_frames = 0
        self._processing_times = []

    # --- stats ---

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def processed_frames(self) -> int:
        return self._processed_frames

    @property
    def average_processing_time_ms(self) -> float:
        if not self._processing_times:
            return 0.0
        return sum(self._processing_times) / len(self._processing_times)
