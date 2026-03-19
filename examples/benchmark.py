#!/usr/bin/env python3
"""Comprehensive synthetic benchmark for tree-trunk-mapper.

Generates forests with varying complexity, point density, noise levels,
and DBH ranges, then evaluates detection precision/recall/F1, position RMSE,
and DBH RMSE.  Also includes a parameter sensitivity analysis.

Usage:
    source /media/sasaki/aiueo/ai_coding_ws/opendata_ws2/.venv/bin/activate
    python examples/benchmark.py
"""

from __future__ import annotations

import csv
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

# Ensure package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tree_trunk_mapper.detector import detect_trunks  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic forest generation
# ---------------------------------------------------------------------------

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
    if rng is None:
        rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, n_points)
    z = rng.uniform(z_min, z_max, n_points)
    x = cx + radius * np.cos(theta) + rng.normal(0, noise_std, n_points)
    y = cy + radius * np.sin(theta) + rng.normal(0, noise_std, n_points)
    return np.column_stack([x, y, z])


@dataclass
class SyntheticForest:
    pcd: o3d.geometry.PointCloud
    true_centers: np.ndarray   # (N, 2) xy positions
    true_radii: np.ndarray     # (N,) radii in metres
    true_dbh: np.ndarray       # (N,) diameters in metres


def generate_forest(
    n_trees: int,
    points_per_tree: int,
    noise_std: float,
    dbh_range: tuple[float, float],
    spacing_factor: float = 2.5,
    seed: int = 0,
) -> SyntheticForest:
    """Generate a synthetic forest point cloud.

    Trees are placed on a regular grid with jitter, with enough spacing to
    avoid overlap.  Ground scatter is added.
    """
    rng = np.random.default_rng(seed)
    max_radius = dbh_range[1] / 2.0

    # Grid layout
    side = int(np.ceil(np.sqrt(n_trees)))
    grid_step = max_radius * 2 * spacing_factor + 1.0
    positions_xy = []
    for i in range(side):
        for j in range(side):
            if len(positions_xy) >= n_trees:
                break
            cx = i * grid_step + rng.uniform(-0.3, 0.3)
            cy = j * grid_step + rng.uniform(-0.3, 0.3)
            positions_xy.append((cx, cy))

    positions_xy = positions_xy[:n_trees]
    radii = rng.uniform(dbh_range[0] / 2.0, dbh_range[1] / 2.0, n_trees)

    all_points = []
    for (cx, cy), r in zip(positions_xy, radii):
        pts = make_cylinder_points(
            cx, cy, r,
            z_min=0.0, z_max=3.0,
            n_points=points_per_tree,
            noise_std=noise_std,
            rng=rng,
        )
        all_points.append(pts)

    # Ground scatter
    x_extent = side * grid_step + 2
    y_extent = side * grid_step + 2
    n_ground = max(200, n_trees * 30)
    ground = np.column_stack([
        rng.uniform(-1, x_extent, n_ground),
        rng.uniform(-1, y_extent, n_ground),
        rng.uniform(-0.05, 0.05, n_ground),
    ])
    all_points.append(ground)

    points = np.vstack(all_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    centers = np.array(positions_xy)
    radii_arr = np.array(radii)
    return SyntheticForest(
        pcd=pcd,
        true_centers=centers,
        true_radii=radii_arr,
        true_dbh=radii_arr * 2.0,
    )


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate(
    forest: SyntheticForest,
    detections: list,
    match_radius: float = 0.5,
) -> dict:
    """Compute precision, recall, F1, position RMSE, and DBH RMSE.

    A detection is matched to the nearest ground-truth trunk within
    *match_radius*.  Each ground-truth trunk can be matched at most once.
    """
    n_gt = len(forest.true_centers)
    n_det = len(detections)

    if n_det == 0:
        return {
            "n_gt": n_gt,
            "n_det": 0,
            "tp": 0,
            "fp": 0,
            "fn": n_gt,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "pos_rmse": float("nan"),
            "dbh_rmse": float("nan"),
        }

    det_xy = np.array([d.center[:2] for d in detections])
    det_dbh = np.array([d.dbh for d in detections])

    gt_tree = KDTree(forest.true_centers)
    matched_gt = set()
    tp_pos_errors = []
    tp_dbh_errors = []

    # Greedy matching: for each detection, find nearest GT
    dists, idxs = gt_tree.query(det_xy)
    # Sort detections by distance to their nearest GT (closest first)
    order = np.argsort(dists)

    tp = 0
    fp = 0
    for i in order:
        gt_idx = idxs[i]
        if dists[i] <= match_radius and gt_idx not in matched_gt:
            matched_gt.add(gt_idx)
            tp += 1
            tp_pos_errors.append(dists[i])
            tp_dbh_errors.append(det_dbh[i] - forest.true_dbh[gt_idx])
        else:
            fp += 1

    fn = n_gt - tp
    precision = tp / n_det if n_det > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    pos_rmse = float(np.sqrt(np.mean(np.array(tp_pos_errors) ** 2))) if tp_pos_errors else float("nan")
    dbh_rmse = float(np.sqrt(np.mean(np.array(tp_dbh_errors) ** 2))) if tp_dbh_errors else float("nan")

    return {
        "n_gt": n_gt,
        "n_det": n_det,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pos_rmse": pos_rmse,
        "dbh_rmse": dbh_rmse,
    }


# ---------------------------------------------------------------------------
# Part 1: Forest complexity benchmark
# ---------------------------------------------------------------------------

def run_forest_benchmark() -> list[dict]:
    """Vary n_trees, points_per_tree, noise, and DBH range."""
    tree_counts = [5, 10, 20, 50]
    point_densities = [500, 1000, 2000, 5000]
    noise_levels = [0.01, 0.02, 0.05]
    dbh_configs = {
        "thin": (0.05, 0.15),
        "medium": (0.10, 0.30),
        "thick": (0.20, 0.50),
    }

    results = []
    total = len(tree_counts) * len(point_densities) * len(noise_levels) * len(dbh_configs)
    count = 0

    for n_trees, pts, noise, (dbh_label, dbh_range) in itertools.product(
        tree_counts, point_densities, noise_levels, dbh_configs.items()
    ):
        count += 1
        print(f"  [{count}/{total}] trees={n_trees}, pts={pts}, noise={noise}, dbh={dbh_label}", end="")

        # Choose parameters adapted to trunk size
        min_r = dbh_range[0] / 2.0
        max_r = dbh_range[1] / 2.0
        eps = max(0.10, min_r * 1.5)
        min_samples = max(5, min(20, pts // 50))

        forest = generate_forest(
            n_trees=n_trees,
            points_per_tree=pts,
            noise_std=noise,
            dbh_range=dbh_range,
            seed=n_trees * 1000 + pts,
        )

        t0 = time.perf_counter()
        detections = detect_trunks(
            forest.pcd,
            eps=eps,
            min_samples=min_samples,
            min_radius=max(0.02, min_r * 0.5),
            max_radius=max_r * 1.5,
            inlier_threshold=max(0.03, noise * 2),
        )
        elapsed = time.perf_counter() - t0

        metrics = evaluate(forest, detections)
        row = {
            "scenario": "forest",
            "n_trees": n_trees,
            "points_per_tree": pts,
            "noise_m": noise,
            "dbh_class": dbh_label,
            "dbh_range": f"{dbh_range[0]}-{dbh_range[1]}",
            "eps": eps,
            "min_samples": min_samples,
            "time_s": round(elapsed, 3),
            **metrics,
        }
        results.append(row)
        print(f"  -> F1={metrics['f1']:.3f}, posRMSE={metrics['pos_rmse']:.4f}, dbhRMSE={metrics['dbh_rmse']:.4f}, {elapsed:.2f}s")

    return results


# ---------------------------------------------------------------------------
# Part 2: Parameter sensitivity analysis
# ---------------------------------------------------------------------------

def run_sensitivity_analysis() -> list[dict]:
    """Vary DBSCAN and slicing parameters on a fixed 20-tree medium forest."""
    eps_values = [0.1, 0.2, 0.3, 0.5]
    min_samples_values = [5, 10, 20, 50]
    slice_thickness_values = [0.1, 0.2, 0.5, 1.0]

    # Fixed reference forest
    forest = generate_forest(
        n_trees=20,
        points_per_tree=2000,
        noise_std=0.02,
        dbh_range=(0.10, 0.30),
        seed=999,
    )

    results = []

    # --- Vary eps ---
    print("\n  Varying eps...")
    for eps in eps_values:
        detections = detect_trunks(forest.pcd, eps=eps, min_samples=10)
        metrics = evaluate(forest, detections)
        results.append({
            "scenario": "sensitivity_eps",
            "varied_param": "eps",
            "param_value": eps,
            "n_trees": 20,
            "points_per_tree": 2000,
            "noise_m": 0.02,
            "dbh_class": "medium",
            "dbh_range": "0.10-0.30",
            "eps": eps,
            "min_samples": 10,
            "time_s": 0,
            **metrics,
        })
        print(f"    eps={eps} -> F1={metrics['f1']:.3f}")

    # --- Vary min_samples ---
    print("  Varying min_samples...")
    for ms in min_samples_values:
        detections = detect_trunks(forest.pcd, eps=0.2, min_samples=ms)
        metrics = evaluate(forest, detections)
        results.append({
            "scenario": "sensitivity_min_samples",
            "varied_param": "min_samples",
            "param_value": ms,
            "n_trees": 20,
            "points_per_tree": 2000,
            "noise_m": 0.02,
            "dbh_class": "medium",
            "dbh_range": "0.10-0.30",
            "eps": 0.2,
            "min_samples": ms,
            "time_s": 0,
            **metrics,
        })
        print(f"    min_samples={ms} -> F1={metrics['f1']:.3f}")

    # --- Vary slice_thickness ---
    print("  Varying slice_thickness...")
    for st in slice_thickness_values:
        detections = detect_trunks(forest.pcd, slice_thickness=st, eps=0.2, min_samples=10)
        metrics = evaluate(forest, detections)
        results.append({
            "scenario": "sensitivity_slice_thickness",
            "varied_param": "slice_thickness",
            "param_value": st,
            "n_trees": 20,
            "points_per_tree": 2000,
            "noise_m": 0.02,
            "dbh_class": "medium",
            "dbh_range": "0.10-0.30",
            "eps": 0.2,
            "min_samples": 10,
            "time_s": 0,
            **metrics,
        })
        print(f"    slice_thickness={st} -> F1={metrics['f1']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "scenario", "n_trees", "points_per_tree", "noise_m", "dbh_class", "dbh_range",
    "eps", "min_samples", "time_s",
    "n_gt", "n_det", "tp", "fp", "fn",
    "precision", "recall", "f1", "pos_rmse", "dbh_rmse",
    "varied_param", "param_value",
]


def save_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Round floats for readability
            out = {}
            for k, v in row.items():
                if isinstance(v, float):
                    out[k] = round(v, 6)
                else:
                    out[k] = v
            writer.writerow(out)


def print_summary_table(title: str, rows: list[dict]) -> None:
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    header = f"{'Trees':>5}  {'Pts':>5}  {'Noise':>6}  {'DBH':>8}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}  {'PosRMSE':>8}  {'DbhRMSE':>8}  {'Time':>6}"
    print(header)
    print("-" * 100)
    for r in rows:
        print(
            f"{r.get('n_trees',''):>5}  "
            f"{r.get('points_per_tree',''):>5}  "
            f"{r.get('noise_m',''):>6}  "
            f"{r.get('dbh_class',''):>8}  "
            f"{r['precision']:>6.3f}  "
            f"{r['recall']:>6.3f}  "
            f"{r['f1']:>6.3f}  "
            f"{r['pos_rmse']:>8.4f}  "
            f"{r['dbh_rmse']:>8.4f}  "
            f"{r.get('time_s',0):>6.2f}"
        )


def print_sensitivity_table(rows: list[dict]) -> None:
    print(f"\n{'=' * 80}")
    print("  Parameter Sensitivity Analysis (20 trees, 2000 pts, noise=0.02, medium DBH)")
    print(f"{'=' * 80}")
    header = f"{'Param':>18}  {'Value':>8}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}  {'PosRMSE':>8}  {'DbhRMSE':>8}"
    print(header)
    print("-" * 80)
    for r in rows:
        print(
            f"{r.get('varied_param',''):>18}  "
            f"{r.get('param_value',''):>8}  "
            f"{r['precision']:>6.3f}  "
            f"{r['recall']:>6.3f}  "
            f"{r['f1']:>6.3f}  "
            f"{r['pos_rmse']:>8.4f}  "
            f"{r['dbh_rmse']:>8.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = Path(__file__).resolve().parent
    csv_path = output_dir / "benchmark_results.csv"

    print("=" * 60)
    print("  tree-trunk-mapper: Comprehensive Synthetic Benchmark")
    print("=" * 60)

    print("\n[1/2] Forest complexity benchmark")
    forest_results = run_forest_benchmark()
    print_summary_table("Forest Complexity Benchmark Results", forest_results)

    print("\n[2/2] Parameter sensitivity analysis")
    sensitivity_results = run_sensitivity_analysis()
    print_sensitivity_table(sensitivity_results)

    # Save all results to CSV
    all_results = forest_results + sensitivity_results
    save_csv(all_results, csv_path)
    print(f"\nResults saved to {csv_path}")

    # Print high-level summary
    forest_only = [r for r in forest_results if r["scenario"] == "forest"]
    if forest_only:
        f1_vals = [r["f1"] for r in forest_only]
        print(f"\n--- Summary ---")
        print(f"  Forest scenarios:   {len(forest_only)}")
        print(f"  Mean F1:            {np.mean(f1_vals):.3f}")
        print(f"  Median F1:          {np.median(f1_vals):.3f}")
        print(f"  Min/Max F1:         {min(f1_vals):.3f} / {max(f1_vals):.3f}")

    # Sensitivity summary: which param matters most?
    print("\n--- Parameter Sensitivity Summary ---")
    for param_name in ["eps", "min_samples", "slice_thickness"]:
        subset = [r for r in sensitivity_results if r.get("varied_param") == param_name]
        if subset:
            f1s = [r["f1"] for r in subset]
            spread = max(f1s) - min(f1s)
            print(f"  {param_name:>20}: F1 range = [{min(f1s):.3f}, {max(f1s):.3f}], spread = {spread:.3f}")


if __name__ == "__main__":
    main()
