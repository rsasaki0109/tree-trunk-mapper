"""Tests for tree_trunk_mapper.export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tree_trunk_mapper.export import export_csv, export_geojson
from tree_trunk_mapper.mapper import TrunkRecord


class TestExportGeoJSON:
    def test_geojson_structure(
        self, tmp_path: Path, sample_trunk_records: list[TrunkRecord]
    ) -> None:
        """Exported GeoJSON should be a valid FeatureCollection."""
        out = tmp_path / "trunks.geojson"
        export_geojson(sample_trunk_records, out)

        data = json.loads(out.read_text())
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 2

        f0 = data["features"][0]
        assert f0["type"] == "Feature"
        assert f0["geometry"]["type"] == "Point"
        assert len(f0["geometry"]["coordinates"]) == 2
        assert f0["properties"]["trunk_id"] == 0
        assert f0["properties"]["species"] == "spruce"
        assert f0["properties"]["dbh"] == pytest.approx(0.30, abs=0.001)

    def test_geojson_coordinates(
        self, tmp_path: Path, sample_trunk_records: list[TrunkRecord]
    ) -> None:
        """Coordinates should match trunk positions."""
        out = tmp_path / "trunks.geojson"
        export_geojson(sample_trunk_records, out)
        data = json.loads(out.read_text())
        coords = data["features"][0]["geometry"]["coordinates"]
        assert coords[0] == pytest.approx(1.0, abs=0.001)
        assert coords[1] == pytest.approx(2.0, abs=0.001)

    def test_empty_export(self, tmp_path: Path) -> None:
        """Exporting empty list should produce valid empty FeatureCollection."""
        out = tmp_path / "empty.geojson"
        export_geojson([], out)
        data = json.loads(out.read_text())
        assert data["type"] == "FeatureCollection"
        assert data["features"] == []


class TestExportCSV:
    def test_csv_contents(
        self, tmp_path: Path, sample_trunk_records: list[TrunkRecord]
    ) -> None:
        """Exported CSV should have correct headers and data."""
        out = tmp_path / "trunks.csv"
        export_csv(sample_trunk_records, out)

        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert set(rows[0].keys()) == {
            "trunk_id", "x", "y", "z", "dbh", "species", "observation_count"
        }
        assert rows[0]["trunk_id"] == "0"
        assert float(rows[0]["x"]) == pytest.approx(1.0, abs=0.001)
        assert rows[0]["species"] == "spruce"
        assert float(rows[0]["dbh"]) == pytest.approx(0.30, abs=0.001)

    def test_empty_csv(self, tmp_path: Path) -> None:
        """Exporting empty list should produce CSV with only header."""
        out = tmp_path / "empty.csv"
        export_csv([], out)
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 0
