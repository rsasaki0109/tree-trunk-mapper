"""Shared fixtures for tree-trunk-mapper tests."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from tree_trunk_mapper.mapper import TrunkRecord


def make_cylinder_points(
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


def points_to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Convert numpy array to Open3D PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


@pytest.fixture()
def single_trunk_pcd() -> o3d.geometry.PointCloud:
    """A point cloud with one cylinder trunk (radius=0.15, center=(1,1))."""
    pts = make_cylinder_points(1.0, 1.0, 0.15, z_min=0.0, z_max=3.0, n_points=1000)
    return points_to_pcd(pts)


@pytest.fixture()
def multi_trunk_pcd() -> o3d.geometry.PointCloud:
    """A point cloud with three well-separated cylinder trunks."""
    rng = np.random.default_rng(42)
    trunks = [
        (0.0, 0.0, 0.10),
        (3.0, 0.0, 0.12),
        (0.0, 3.0, 0.20),
    ]
    all_pts = []
    for cx, cy, r in trunks:
        all_pts.append(make_cylinder_points(cx, cy, r, n_points=800, rng=rng))
    # Add some ground scatter
    ground = rng.uniform([-2, -2, 0.0], [5, 5, 0.05], (200, 3))
    all_pts.append(ground)
    return points_to_pcd(np.vstack(all_pts))


@pytest.fixture()
def sample_trunk_records() -> list[TrunkRecord]:
    """A list of sample TrunkRecord objects for export tests."""
    return [
        TrunkRecord(
            trunk_id=0,
            position=np.array([1.0, 2.0, 1.3]),
            dbh=0.30,
            observation_count=3,
            species="spruce",
        ),
        TrunkRecord(
            trunk_id=1,
            position=np.array([5.0, 6.0, 1.4]),
            dbh=0.24,
            observation_count=1,
            species="unknown",
        ),
    ]
