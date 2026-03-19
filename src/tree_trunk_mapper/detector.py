"""Tree trunk detection: height slicing, DBSCAN clustering, cylinder fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


@dataclass
class TrunkDetection:
    """A single detected tree trunk."""

    center: np.ndarray  # (3,) position of the trunk centre at breast height
    radius: float  # estimated trunk radius (metres)
    dbh: float  # diameter at breast height (metres)
    inlier_count: int  # number of inlier points in the cylinder fit
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def to_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "radius": float(self.radius),
            "dbh": float(self.dbh),
            "inlier_count": int(self.inlier_count),
            "axis": self.axis.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrunkDetection:
        return cls(
            center=np.array(d["center"]),
            radius=d["radius"],
            dbh=d["dbh"],
            inlier_count=d["inlier_count"],
            axis=np.array(d.get("axis", [0.0, 0.0, 1.0])),
        )


def height_slice(
    pcd: o3d.geometry.PointCloud,
    slice_height: float = 1.3,
    slice_thickness: float = 0.2,
    ground_z: float | None = None,
) -> o3d.geometry.PointCloud:
    """Extract a horizontal slice at breast height.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Input point cloud.
    slice_height : float
        Height above ground to centre the slice (default 1.3 m = breast height).
    slice_thickness : float
        Total thickness of the slice.
    ground_z : float or None
        If None, the ground level is estimated as the 5th percentile of z values.

    Returns
    -------
    o3d.geometry.PointCloud
        Points within the height slice.
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return o3d.geometry.PointCloud()

    if ground_z is None:
        ground_z = np.percentile(points[:, 2], 5)

    z_min = ground_z + slice_height - slice_thickness / 2
    z_max = ground_z + slice_height + slice_thickness / 2

    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    indices = np.where(mask)[0]

    return pcd.select_by_index(indices.tolist())


def cluster_points(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.15,
    min_samples: int = 20,
) -> list[np.ndarray]:
    """Cluster 2D (xy) points using DBSCAN.

    Returns
    -------
    list[np.ndarray]
        List of (N_i, 3) arrays, one per cluster.
    """
    points = np.asarray(pcd.points)
    if len(points) < min_samples:
        return []

    xy = points[:, :2]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy)

    clusters = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_mask = labels == label
        clusters.append(points[cluster_mask])

    return clusters


def fit_cylinder_ransac(
    points: np.ndarray,
    n_iterations: int = 1000,
    distance_threshold: float = 0.03,
    min_radius: float = 0.02,
    max_radius: float = 0.50,
) -> tuple[np.ndarray, float, int] | None:
    """Fit a vertical cylinder to a set of 3D points using RANSAC.

    The cylinder is assumed to have a vertical axis (parallel to z).
    We fit a circle in the xy-plane.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) point array.
    n_iterations : int
        Number of RANSAC iterations.
    distance_threshold : float
        Max distance from the circle boundary to count as inlier.
    min_radius, max_radius : float
        Valid radius range for tree trunks.

    Returns
    -------
    (center, radius, inlier_count) or None if fitting fails.
    """
    xy = points[:, :2]
    n = len(xy)
    if n < 3:
        return None

    best_inliers = 0
    best_center = None
    best_radius = 0.0

    rng = np.random.default_rng(42)

    for _ in range(n_iterations):
        # Sample 3 points to define a circle
        idx = rng.choice(n, size=3, replace=False)
        p1, p2, p3 = xy[idx]

        # Solve for circle through 3 points
        center, radius = _circle_from_three_points(p1, p2, p3)
        if center is None:
            continue
        if radius < min_radius or radius > max_radius:
            continue

        # Count inliers: points whose distance to the circle boundary is small
        dists = np.abs(np.linalg.norm(xy - center, axis=1) - radius)
        inlier_mask = dists < distance_threshold
        inlier_count = int(inlier_mask.sum())

        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_center = center
            best_radius = radius

    if best_center is None or best_inliers < 3:
        return None

    # Refine: least-squares circle fit on inliers
    dists = np.abs(np.linalg.norm(xy - best_center, axis=1) - best_radius)
    inlier_mask = dists < distance_threshold
    inlier_xy = xy[inlier_mask]

    if len(inlier_xy) >= 3:
        refined_center, refined_radius = _least_squares_circle(inlier_xy)
        if refined_center is not None and min_radius <= refined_radius <= max_radius:
            best_center = refined_center
            best_radius = refined_radius
            dists = np.abs(np.linalg.norm(xy - best_center, axis=1) - best_radius)
            best_inliers = int((dists < distance_threshold).sum())

    center_3d = np.array([best_center[0], best_center[1], float(np.mean(points[:, 2]))])
    return center_3d, best_radius, best_inliers


def _circle_from_three_points(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> tuple[np.ndarray | None, float]:
    """Compute the circumscribed circle of three 2D points."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3

    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-10:
        return None, 0.0

    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d

    center = np.array([ux, uy])
    radius = float(np.linalg.norm(p1 - center))
    return center, radius


def _least_squares_circle(points: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Algebraic least-squares circle fit (Kasa method).

    Parameters
    ----------
    points : np.ndarray
        (N, 2) array of xy coordinates.

    Returns
    -------
    (center, radius) or (None, 0.0) on failure.
    """
    x = points[:, 0]
    y = points[:, 1]
    n = len(x)
    if n < 3:
        return None, 0.0

    # Solve: [x, y, 1] @ [A, B, C]^T = x^2 + y^2
    A_mat = np.column_stack([x, y, np.ones(n)])
    b_vec = x * x + y * y

    try:
        result, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    except np.linalg.LinAlgError:
        return None, 0.0

    cx = result[0] / 2.0
    cy = result[1] / 2.0
    r_sq = result[2] + cx * cx + cy * cy
    if r_sq < 0:
        return None, 0.0

    return np.array([cx, cy]), float(np.sqrt(r_sq))


def detect_trunks(
    pcd: o3d.geometry.PointCloud,
    slice_height: float = 1.3,
    slice_thickness: float = 0.2,
    eps: float = 0.15,
    min_samples: int = 20,
    ransac_iterations: int = 1000,
    inlier_threshold: float = 0.03,
    min_radius: float = 0.02,
    max_radius: float = 0.50,
    min_inlier_ratio: float = 0.4,
) -> list[TrunkDetection]:
    """Full detection pipeline: height slice -> DBSCAN -> cylinder fit.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Input point cloud (single frame).
    slice_height, slice_thickness : float
        Height-slice parameters.
    eps, min_samples : float, int
        DBSCAN parameters.
    ransac_iterations : int
        RANSAC iterations for cylinder fitting.
    inlier_threshold : float
        Distance threshold for RANSAC inliers.
    min_radius, max_radius : float
        Valid trunk radius range (metres).
    min_inlier_ratio : float
        Minimum ratio of inliers to total cluster points to accept a detection.

    Returns
    -------
    list[TrunkDetection]
    """
    # Step 1: height slice at breast height
    sliced = height_slice(pcd, slice_height=slice_height, slice_thickness=slice_thickness)

    # Step 2: DBSCAN clustering
    clusters = cluster_points(sliced, eps=eps, min_samples=min_samples)

    # Step 3: cylinder (circle) fitting per cluster
    detections: list[TrunkDetection] = []
    for cluster_pts in clusters:
        result = fit_cylinder_ransac(
            cluster_pts,
            n_iterations=ransac_iterations,
            distance_threshold=inlier_threshold,
            min_radius=min_radius,
            max_radius=max_radius,
        )
        if result is None:
            continue

        center, radius, inlier_count = result
        inlier_ratio = inlier_count / len(cluster_pts)
        if inlier_ratio < min_inlier_ratio:
            continue

        detections.append(TrunkDetection(
            center=center,
            radius=radius,
            dbh=2.0 * radius,
            inlier_count=inlier_count,
        ))

    return detections
