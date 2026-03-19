"""Export trunk map as GeoJSON or CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tree_trunk_mapper.mapper import TrunkRecord


def export_geojson(trunks: list[TrunkRecord], output_path: Path) -> None:
    """Export trunk map as a GeoJSON FeatureCollection.

    Each trunk is represented as a Point feature with properties:
    trunk_id, species, dbh, observation_count, and z coordinate.
    Coordinates use the point cloud's local coordinate system (x, y).
    """
    features = []
    for trunk in trunks:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    round(float(trunk.position[0]), 4),
                    round(float(trunk.position[1]), 4),
                ],
            },
            "properties": {
                "trunk_id": trunk.trunk_id,
                "species": trunk.species,
                "dbh": round(trunk.dbh, 4),
                "z": round(float(trunk.position[2]), 4),
                "observation_count": trunk.observation_count,
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    Path(output_path).write_text(json.dumps(geojson, indent=2))


def export_csv(trunks: list[TrunkRecord], output_path: Path) -> None:
    """Export trunk map as a CSV file."""
    fieldnames = ["trunk_id", "x", "y", "z", "dbh", "species", "observation_count"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trunk in trunks:
            writer.writerow({
                "trunk_id": trunk.trunk_id,
                "x": round(float(trunk.position[0]), 4),
                "y": round(float(trunk.position[1]), 4),
                "z": round(float(trunk.position[2]), 4),
                "dbh": round(trunk.dbh, 4),
                "species": trunk.species,
                "observation_count": trunk.observation_count,
            })
