"""Tests for the streaming detection module."""

from __future__ import annotations

import numpy as np
import pytest

from tree_trunk_mapper.detector import TrunkDetection
from tree_trunk_mapper.streaming import StreamingDetector, StreamingResult


def _make_cylinder_points(
    cx: float,
    cy: float,
    radius: float,
    z_min: float = 0.0,
    z_max: float = 3.0,
    n_points: int = 500,
    noise_std: float = 0.005,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate noisy points on a vertical cylinder surface."""
    if rng is None:
        rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, n_points)
    z = rng.uniform(z_min, z_max, n_points)
    x = cx + radius * np.cos(theta) + rng.normal(0, noise_std, n_points)
    y = cy + radius * np.sin(theta) + rng.normal(0, noise_std, n_points)
    return np.column_stack([x, y, z])


def _make_frame(trunks: list[tuple[float, float, float]], rng_seed: int = 0) -> np.ndarray:
    """Build a synthetic frame with the given trunks (cx, cy, radius)."""
    rng = np.random.default_rng(rng_seed)
    parts = []
    for cx, cy, r in trunks:
        parts.append(_make_cylinder_points(cx, cy, r, n_points=3000, rng=rng))
    # Add ground scatter
    ground = rng.uniform([-2, -2, 0.0], [5, 5, 0.05], (200, 3))
    parts.append(ground)
    return np.vstack(parts)


class TestStreamingDetector:
    """Tests for StreamingDetector."""

    def test_single_frame(self) -> None:
        detector = StreamingDetector()
        frame = _make_frame([(1.0, 1.0, 0.15)])
        result = detector.process_frame(frame)

        assert isinstance(result, StreamingResult)
        assert result.frame_count == 1
        assert result.total_trunks >= 1
        assert result.processing_time_ms > 0
        assert len(result.new_detections) >= 1

    def test_multi_frame_accumulation(self) -> None:
        detector = StreamingDetector(merge_radius=0.5)

        # Frame 1: trunk at (1, 1)
        frame1 = _make_frame([(1.0, 1.0, 0.15)], rng_seed=0)
        r1 = detector.process_frame(frame1)
        assert r1.total_trunks >= 1

        # Frame 2: same trunk at (1, 1) + new trunk at (5, 5)
        frame2 = _make_frame([(1.0, 1.0, 0.15), (5.0, 5.0, 0.12)], rng_seed=1)
        r2 = detector.process_frame(frame2)
        assert r2.total_trunks >= 2
        assert r2.merged_count >= 1  # trunk at (1,1) should merge

    def test_frame_skipping(self) -> None:
        detector = StreamingDetector(process_every_n=3)
        frame = _make_frame([(1.0, 1.0, 0.15)])

        r1 = detector.process_frame(frame)  # frame 1 - skipped
        assert r1.new_detections == []
        assert r1.processing_time_ms == 0.0

        r2 = detector.process_frame(frame)  # frame 2 - skipped
        assert r2.new_detections == []

        r3 = detector.process_frame(frame)  # frame 3 - processed
        assert len(r3.new_detections) >= 1
        assert r3.processing_time_ms > 0

        assert detector.total_frames == 3
        assert detector.processed_frames == 1

    def test_reset(self) -> None:
        detector = StreamingDetector()
        frame = _make_frame([(1.0, 1.0, 0.15)])
        detector.process_frame(frame)

        assert detector.total_frames == 1
        assert len(detector.get_map()) >= 1

        detector.reset()

        assert detector.total_frames == 0
        assert detector.processed_frames == 0
        assert len(detector.get_map()) == 0
        assert detector.average_processing_time_ms == 0.0

    def test_processing_time_tracking(self) -> None:
        detector = StreamingDetector()
        frame = _make_frame([(1.0, 1.0, 0.15)])

        detector.process_frame(frame)
        detector.process_frame(frame)

        assert detector.processed_frames == 2
        assert detector.average_processing_time_ms > 0

    def test_callback(self) -> None:
        received: list[TrunkDetection] = []

        def on_det(det: TrunkDetection) -> None:
            received.append(det)

        detector = StreamingDetector(on_detection=on_det)
        frame = _make_frame([(1.0, 1.0, 0.15)])
        detector.process_frame(frame)

        assert len(received) >= 1
        assert all(isinstance(d, TrunkDetection) for d in received)

    def test_get_map(self) -> None:
        detector = StreamingDetector()
        assert detector.get_map() == []

        frame = _make_frame([(2.0, 2.0, 0.10)])
        detector.process_frame(frame)

        trunk_map = detector.get_map()
        assert len(trunk_map) >= 1
        assert trunk_map[0].trunk_id == 0

    def test_invalid_process_every_n(self) -> None:
        with pytest.raises(ValueError, match="process_every_n must be >= 1"):
            StreamingDetector(process_every_n=0)
