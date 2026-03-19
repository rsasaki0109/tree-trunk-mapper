"""Tests for ROS2 conversion utilities."""

from __future__ import annotations

import numpy as np
import pytest

from tree_trunk_mapper.mapper import TrunkRecord

# Skip the entire module if rclpy is not installed
rclpy = pytest.importorskip("rclpy", reason="rclpy not installed, skipping ROS2 tests")


from ros2_node.conversions import (  # noqa: E402
    trunks_to_marker_array,
    trunks_to_pose_array,
)
from visualization_msgs.msg import Marker  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402


@pytest.fixture()
def sample_trunks() -> list[TrunkRecord]:
    return [
        TrunkRecord(trunk_id=0, position=np.array([1.0, 2.0, 1.3]), dbh=0.30),
        TrunkRecord(trunk_id=1, position=np.array([5.0, 6.0, 1.4]), dbh=0.24),
        TrunkRecord(trunk_id=2, position=np.array([3.0, 4.0, 1.2]), dbh=0.18),
    ]


@pytest.fixture()
def stamp() -> Time:
    t = Time()
    t.sec = 100
    t.nanosec = 0
    return t


class TestTrunksToMarkerArray:
    def test_marker_count(self, sample_trunks, stamp):
        """MarkerArray should have N+1 markers (1 DELETE_ALL + N cylinders)."""
        result = trunks_to_marker_array(sample_trunks, "map", stamp)
        # 1 DELETE_ALL + 3 trunk markers
        assert len(result.markers) == 4

    def test_delete_all_first(self, sample_trunks, stamp):
        result = trunks_to_marker_array(sample_trunks, "map", stamp)
        assert result.markers[0].action == Marker.DELETEALL

    def test_cylinder_type(self, sample_trunks, stamp):
        result = trunks_to_marker_array(sample_trunks, "map", stamp)
        for marker in result.markers[1:]:
            assert marker.type == Marker.CYLINDER

    def test_marker_positions(self, sample_trunks, stamp):
        result = trunks_to_marker_array(sample_trunks, "map", stamp)
        for i, trunk in enumerate(sample_trunks):
            marker = result.markers[i + 1]
            assert marker.pose.position.x == pytest.approx(trunk.position[0])
            assert marker.pose.position.y == pytest.approx(trunk.position[1])

    def test_marker_scale_matches_dbh(self, sample_trunks, stamp):
        result = trunks_to_marker_array(sample_trunks, "map", stamp)
        for i, trunk in enumerate(sample_trunks):
            marker = result.markers[i + 1]
            assert marker.scale.x == pytest.approx(trunk.dbh)
            assert marker.scale.y == pytest.approx(trunk.dbh)

    def test_frame_id(self, sample_trunks, stamp):
        result = trunks_to_marker_array(sample_trunks, "odom", stamp)
        for marker in result.markers:
            assert marker.header.frame_id == "odom"

    def test_empty_trunks(self, stamp):
        result = trunks_to_marker_array([], "map", stamp)
        # Only the DELETE_ALL marker
        assert len(result.markers) == 1
        assert result.markers[0].action == Marker.DELETEALL


class TestTrunksToPoseArray:
    def test_pose_count(self, sample_trunks, stamp):
        result = trunks_to_pose_array(sample_trunks, "map", stamp)
        assert len(result.poses) == 3

    def test_pose_positions(self, sample_trunks, stamp):
        result = trunks_to_pose_array(sample_trunks, "map", stamp)
        for i, trunk in enumerate(sample_trunks):
            pose = result.poses[i]
            assert pose.position.x == pytest.approx(trunk.position[0])
            assert pose.position.y == pytest.approx(trunk.position[1])
            assert pose.position.z == pytest.approx(trunk.position[2])

    def test_frame_id(self, sample_trunks, stamp):
        result = trunks_to_pose_array(sample_trunks, "base_link", stamp)
        assert result.header.frame_id == "base_link"

    def test_empty_trunks(self, stamp):
        result = trunks_to_pose_array([], "map", stamp)
        assert len(result.poses) == 0
