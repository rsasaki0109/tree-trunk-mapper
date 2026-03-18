"""Tests for tree_trunk_mapper.detector."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from tree_trunk_mapper.detector import (
    TrunkDetection,
    cluster_points,
    detect_trunks,
    fit_cylinder_ransac,
    height_slice,
)

from .conftest import make_cylinder_points, points_to_pcd


# ---------- height_slice ----------

class TestHeightSlice:
    def test_basic_slice(self, single_trunk_pcd: o3d.geometry.PointCloud) -> None:
        """Slicing should return only points near breast height."""
        sliced = height_slice(single_trunk_pcd, slice_height=1.3, slice_thickness=0.2, ground_z=0.0)
        pts = np.asarray(sliced.points)
        assert len(pts) > 0
        assert pts[:, 2].min() >= 1.19
        assert pts[:, 2].max() <= 1.41

    def test_empty_input(self) -> None:
        """Slicing an empty cloud should return an empty cloud."""
        pcd = o3d.geometry.PointCloud()
        result = height_slice(pcd)
        assert len(np.asarray(result.points)) == 0

    def test_ground_z_auto(self) -> None:
        """When ground_z is None, it should be estimated from the 5th percentile."""
        pts = make_cylinder_points(0, 0, 0.1, z_min=10.0, z_max=13.0, n_points=500)
        pcd = points_to_pcd(pts)
        sliced = height_slice(pcd, slice_height=1.3, slice_thickness=0.2, ground_z=None)
        pts_out = np.asarray(sliced.points)
        # Ground should be estimated ~10.0, so slice should be around 11.2-11.4
        assert len(pts_out) > 0
        assert pts_out[:, 2].min() >= 11.0
        assert pts_out[:, 2].max() <= 11.7

    def test_no_points_in_slice(self) -> None:
        """If no points fall in the slice range, return empty."""
        pts = np.array([[0, 0, 0], [0, 0, 0.1], [0, 0, 0.2]], dtype=np.float64)
        pcd = points_to_pcd(pts)
        sliced = height_slice(pcd, slice_height=5.0, slice_thickness=0.1, ground_z=0.0)
        assert len(np.asarray(sliced.points)) == 0


# ---------- cluster_points ----------

class TestClusterPoints:
    def test_two_clusters(self) -> None:
        """Two well-separated blobs should produce two clusters."""
        rng = np.random.default_rng(0)
        blob1 = rng.normal([0, 0, 1.3], 0.02, (50, 3))
        blob2 = rng.normal([5, 5, 1.3], 0.02, (50, 3))
        pts = np.vstack([blob1, blob2])
        pcd = points_to_pcd(pts)
        clusters = cluster_points(pcd, eps=0.15, min_samples=10)
        assert len(clusters) == 2

    def test_too_few_points(self) -> None:
        """Fewer points than min_samples should return no clusters."""
        pts = np.array([[0, 0, 1.3], [0.01, 0.01, 1.3]], dtype=np.float64)
        pcd = points_to_pcd(pts)
        clusters = cluster_points(pcd, eps=0.15, min_samples=20)
        assert clusters == []

    def test_single_cluster(self) -> None:
        """One dense blob should produce one cluster."""
        rng = np.random.default_rng(1)
        pts = rng.normal([0, 0, 1.3], 0.02, (100, 3))
        pcd = points_to_pcd(pts)
        clusters = cluster_points(pcd, eps=0.15, min_samples=10)
        assert len(clusters) == 1
        assert len(clusters[0]) >= 90  # almost all points


# ---------- fit_cylinder_ransac ----------

class TestFitCylinderRansac:
    def test_perfect_circle(self) -> None:
        """Points on a perfect circle should be fitted accurately."""
        theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        r = 0.15
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.full_like(x, 1.3)
        pts = np.column_stack([x, y, z])

        result = fit_cylinder_ransac(pts)
        assert result is not None
        center, radius, inlier_count = result
        assert radius == pytest.approx(r, abs=0.01)
        assert center[0] == pytest.approx(0.0, abs=0.01)
        assert center[1] == pytest.approx(0.0, abs=0.01)
        assert inlier_count >= 180

    def test_noisy_circle(self) -> None:
        """Noisy cylinder points should still be fitted reasonably."""
        pts = make_cylinder_points(2.0, 3.0, 0.12, z_min=1.2, z_max=1.4, n_points=300)
        result = fit_cylinder_ransac(pts)
        assert result is not None
        center, radius, inlier_count = result
        assert radius == pytest.approx(0.12, abs=0.03)
        assert center[0] == pytest.approx(2.0, abs=0.05)
        assert center[1] == pytest.approx(3.0, abs=0.05)

    def test_too_few_points(self) -> None:
        """Fewer than 3 points should return None."""
        pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        assert fit_cylinder_ransac(pts) is None

    def test_radius_out_of_range(self) -> None:
        """Points forming a huge circle should be rejected by default bounds."""
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        r = 2.0  # way beyond max_radius=0.50
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), np.full(100, 1.3)])
        result = fit_cylinder_ransac(pts, max_radius=0.50)
        assert result is None


# ---------- detect_trunks (integration) ----------

class TestDetectTrunks:
    def test_single_trunk(self, single_trunk_pcd: o3d.geometry.PointCloud) -> None:
        """Should detect exactly one trunk from a single-cylinder cloud."""
        detections = detect_trunks(single_trunk_pcd, min_samples=10)
        assert len(detections) == 1
        d = detections[0]
        assert isinstance(d, TrunkDetection)
        assert d.radius == pytest.approx(0.15, abs=0.03)
        assert d.dbh == pytest.approx(0.30, abs=0.06)
        assert d.center[0] == pytest.approx(1.0, abs=0.1)
        assert d.center[1] == pytest.approx(1.0, abs=0.1)

    def test_multi_trunk(self, multi_trunk_pcd: o3d.geometry.PointCloud) -> None:
        """Should detect three trunks from a multi-cylinder cloud."""
        detections = detect_trunks(multi_trunk_pcd, min_samples=10, eps=0.3)
        assert len(detections) == 3

    def test_empty_cloud(self) -> None:
        """Empty cloud should return no detections."""
        pcd = o3d.geometry.PointCloud()
        detections = detect_trunks(pcd)
        assert detections == []

    def test_trunk_detection_to_dict_roundtrip(self) -> None:
        """TrunkDetection should survive dict serialization roundtrip."""
        d = TrunkDetection(
            center=np.array([1.0, 2.0, 1.3]),
            radius=0.15,
            dbh=0.30,
            inlier_count=100,
        )
        d2 = TrunkDetection.from_dict(d.to_dict())
        np.testing.assert_allclose(d2.center, d.center)
        assert d2.radius == d.radius
        assert d2.dbh == d.dbh
        assert d2.inlier_count == d.inlier_count
