"""Tests for tree_trunk_mapper.cli."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import open3d as o3d
from click.testing import CliRunner

from tree_trunk_mapper.cli import cli
from tree_trunk_mapper.mapper import TrunkRecord

from .conftest import make_cylinder_points, points_to_pcd


def _write_npy_cloud(path: Path, n_trunks: int = 1) -> None:
    """Write a synthetic point cloud as .npy for testing."""
    rng = np.random.default_rng(42)
    all_pts = []
    for i in range(n_trunks):
        pts = make_cylinder_points(
            cx=i * 5.0, cy=0.0, radius=0.12,
            z_min=0.0, z_max=3.0, n_points=500, rng=rng,
        )
        all_pts.append(pts)
    # Ground scatter
    ground = rng.uniform([-1, -1, 0.0], [n_trunks * 5.0, 1, 0.05], (200, 3))
    all_pts.append(ground)
    np.save(str(path), np.vstack(all_pts))


class TestCLIDetect:
    def test_detect_single_file(self, tmp_path: Path) -> None:
        """CLI detect should produce a JSON detections file."""
        npy_path = tmp_path / "scan.npy"
        _write_npy_cloud(npy_path, n_trunks=1)

        runner = CliRunner()
        output_path = tmp_path / "detections.json"
        result = runner.invoke(cli, [
            "detect", str(npy_path),
            "--min-samples", "10",
            "-o", str(output_path),
        ])
        assert result.exit_code == 0, result.output
        assert output_path.exists()

        data = json.loads(output_path.read_text())
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_detect_default_output(self, tmp_path: Path) -> None:
        """Without -o, detect should create <input>_detections.json."""
        npy_path = tmp_path / "scan.npy"
        _write_npy_cloud(npy_path)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "detect", str(npy_path),
            "--min-samples", "10",
        ])
        assert result.exit_code == 0, result.output
        default_out = tmp_path / "scan_detections.json"
        assert default_out.exists()


class TestCLIExport:
    def test_export_geojson(self, tmp_path: Path) -> None:
        """CLI export should produce a GeoJSON file."""
        # Write a map JSON first
        records = [
            TrunkRecord(
                trunk_id=0,
                position=np.array([1.0, 2.0, 1.3]),
                dbh=0.30,
                observation_count=1,
            ).to_dict()
        ]
        map_path = tmp_path / "trunk_map.json"
        map_path.write_text(json.dumps(records))

        runner = CliRunner()
        out = tmp_path / "out.geojson"
        result = runner.invoke(cli, [
            "export", str(map_path), "-f", "geojson", "-o", str(out),
        ])
        assert result.exit_code == 0, result.output
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["type"] == "FeatureCollection"

    def test_export_csv(self, tmp_path: Path) -> None:
        """CLI export should produce a CSV file."""
        records = [
            TrunkRecord(
                trunk_id=0,
                position=np.array([1.0, 2.0, 1.3]),
                dbh=0.30,
                observation_count=1,
            ).to_dict()
        ]
        map_path = tmp_path / "trunk_map.json"
        map_path.write_text(json.dumps(records))

        runner = CliRunner()
        out = tmp_path / "out.csv"
        result = runner.invoke(cli, [
            "export", str(map_path), "-f", "csv", "-o", str(out),
        ])
        assert result.exit_code == 0, result.output
        assert out.exists()


class TestCLIVersion:
    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output
