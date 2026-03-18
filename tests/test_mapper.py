"""Tests for tree_trunk_mapper.mapper."""

from __future__ import annotations

import numpy as np
import pytest

from tree_trunk_mapper.detector import TrunkDetection
from tree_trunk_mapper.mapper import TrunkMapper, TrunkRecord


def _make_detection(cx: float, cy: float, radius: float = 0.10) -> TrunkDetection:
    return TrunkDetection(
        center=np.array([cx, cy, 1.3]),
        radius=radius,
        dbh=2 * radius,
        inlier_count=50,
    )


class TestTrunkMapper:
    def test_single_detection(self) -> None:
        """A single detection should create one trunk record."""
        mapper = TrunkMapper(merge_radius=0.5)
        mapper.add_detections([_make_detection(1.0, 2.0)])
        trunk_map = mapper.get_map()
        assert len(trunk_map) == 1
        assert trunk_map[0].trunk_id == 0
        assert trunk_map[0].observation_count == 1
        np.testing.assert_allclose(trunk_map[0].position[:2], [1.0, 2.0], atol=0.01)

    def test_merge_nearby(self) -> None:
        """Two detections within merge_radius should be merged."""
        mapper = TrunkMapper(merge_radius=0.5)
        mapper.add_detections([_make_detection(1.0, 2.0, 0.10)])
        mapper.add_detections([_make_detection(1.1, 2.1, 0.12)])
        trunk_map = mapper.get_map()
        assert len(trunk_map) == 1
        assert trunk_map[0].observation_count == 2
        # Position should be the mean
        np.testing.assert_allclose(trunk_map[0].position[:2], [1.05, 2.05], atol=0.01)
        # DBH should be the mean
        assert trunk_map[0].dbh == pytest.approx(0.22, abs=0.01)

    def test_separate_trunks(self) -> None:
        """Two detections far apart should create two separate trunks."""
        mapper = TrunkMapper(merge_radius=0.5)
        mapper.add_detections([_make_detection(0.0, 0.0)])
        mapper.add_detections([_make_detection(10.0, 10.0)])
        trunk_map = mapper.get_map()
        assert len(trunk_map) == 2

    def test_empty_detections(self) -> None:
        """Adding empty detections should not change the map."""
        mapper = TrunkMapper()
        mapper.add_detections([])
        assert mapper.get_map() == []

    def test_multiple_merges(self) -> None:
        """Three observations of the same trunk should merge correctly."""
        mapper = TrunkMapper(merge_radius=0.5)
        mapper.add_detections([_make_detection(1.0, 1.0, 0.10)])
        mapper.add_detections([_make_detection(1.0, 1.0, 0.12)])
        mapper.add_detections([_make_detection(1.0, 1.0, 0.14)])
        trunk_map = mapper.get_map()
        assert len(trunk_map) == 1
        assert trunk_map[0].observation_count == 3
        assert trunk_map[0].dbh == pytest.approx(0.24, abs=0.01)  # mean of 0.20, 0.24, 0.28

    def test_batch_detections(self) -> None:
        """Multiple detections in a single frame should all be added."""
        mapper = TrunkMapper(merge_radius=0.5)
        dets = [
            _make_detection(0, 0),
            _make_detection(5, 5),
            _make_detection(10, 10),
        ]
        mapper.add_detections(dets)
        assert len(mapper.get_map()) == 3

    def test_trunk_record_roundtrip(self) -> None:
        """TrunkRecord should survive dict serialization roundtrip."""
        rec = TrunkRecord(
            trunk_id=7,
            position=np.array([1.0, 2.0, 1.3]),
            dbh=0.30,
            observation_count=5,
            species="birch",
        )
        rec2 = TrunkRecord.from_dict(rec.to_dict())
        assert rec2.trunk_id == rec.trunk_id
        np.testing.assert_allclose(rec2.position, rec.position)
        assert rec2.dbh == rec.dbh
        assert rec2.observation_count == rec.observation_count
        assert rec2.species == rec.species
