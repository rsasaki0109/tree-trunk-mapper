# tree-trunk-mapper

Detect and map individual tree trunks from LiDAR point clouds, designed for the [FinnWoodlands](https://www.finnwoodlands.fi/) dataset (Ouster OS1 + ZED2 stereo, backpack-mounted, with Spruce/Birch/Pine annotations).

## Pipeline

1. **Height slicing** -- Extract a horizontal band at breast height (~1.3 m) to isolate trunk cross-sections.
2. **DBSCAN clustering** -- Cluster the 2D (xy) points to separate individual trunks.
3. **Cylinder fitting (RANSAC)** -- Fit a circle/cylinder to each cluster to estimate trunk centre and DBH (diameter at breast height).
4. **Multi-frame mapping** -- Accumulate detections across frames, merging nearby observations into a persistent trunk map.

## Installation

```bash
pip install -e .
```

## Usage

### Detect trunks in a single frame

```bash
tree-trunk-mapper detect scan.pcd
```

### Build a trunk map from a sequence

```bash
tree-trunk-mapper map scans/ --glob-pattern "*.pcd"
```

### Export as GeoJSON or CSV

```bash
tree-trunk-mapper export trunk_map.json -f geojson -o trunks.geojson
tree-trunk-mapper export trunk_map.json -f csv -o trunks.csv
```

### Key options

| Option | Default | Description |
|--------|---------|-------------|
| `--slice-height` | 1.3 | Height above ground for breast-height slice (m) |
| `--slice-thickness` | 0.2 | Thickness of the height slice (m) |
| `--eps` | 0.15 | DBSCAN epsilon (m) |
| `--min-samples` | 20 | DBSCAN minimum points per cluster |
| `--merge-radius` | 0.5 | Radius to merge detections across frames (m) |

## License

MIT
