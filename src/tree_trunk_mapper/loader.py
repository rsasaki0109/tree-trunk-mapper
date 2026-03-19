"""Load point cloud data from various formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    """Load a point cloud file.

    Supports PCD, PLY, LAS/LAZ (via numpy fallback), and NPY formats.

    Parameters
    ----------
    path : Path
        Path to the point cloud file.

    Returns
    -------
    o3d.geometry.PointCloud
        The loaded point cloud.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".pcd", ".ply", ".xyz", ".xyzn", ".pts"):
        pcd = o3d.io.read_point_cloud(str(path))
        if pcd.is_empty():
            raise ValueError(f"Failed to load point cloud from {path}")
        return pcd

    if suffix == ".npy":
        points = np.load(str(path))
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected (N, 3+) array, got shape {points.shape}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        return pcd

    if suffix == ".bin":
        # KITTI-style binary format: float32, 4 columns (x, y, z, intensity)
        points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        return pcd

    # Fallback: let Open3D try
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Unsupported or empty point cloud file: {path}")
    return pcd
