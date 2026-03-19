# tree-trunk-mapper

[![CI](https://github.com/rsasaki0109/tree-trunk-mapper/actions/workflows/ci.yml/badge.svg)](https://github.com/rsasaki0109/tree-trunk-mapper/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Detect and map individual tree trunks from LiDAR point clouds, designed for the [FinnWoodlands](https://www.finnwoodlands.fi/) dataset (Ouster OS1 + ZED2 stereo, backpack-mounted, with Spruce/Birch/Pine annotations).

## Algorithm

The detection pipeline follows three stages:

```
Raw Point Cloud
      |
      v
[1] Height Slice (1.3 m +/- 0.1 m)
      |  Extract a horizontal band at breast height to isolate
      |  trunk cross-sections, filtering out canopy and ground.
      v
[2] DBSCAN Clustering (eps=0.15 m, min_samples=20)
      |  Cluster the 2D (xy) points so that each cluster
      |  corresponds to an individual trunk cross-section.
      v
[3] RANSAC Cylinder Fitting
      |  Fit a circle in the xy-plane to each cluster using
      |  RANSAC (1000 iterations), then refine with least-squares.
      |  Reject fits outside the valid trunk radius range (2-50 cm).
      v
Trunk Detections (centre position + DBH)
```

**Multi-frame mapping:** When processing a sequence of scans, detections are accumulated in a `TrunkMapper` that merges observations within a configurable radius (default 0.5 m) using incremental mean updates for position and DBH.

## Installation

```bash
pip install -e .
```

For development (pytest + ruff):

```bash
pip install -e ".[dev]"
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

### Streaming / Real-time Mode

Stream-process LiDAR frames as they arrive, with incremental map updates.
Unlike batch processing, streaming mode detects trunks frame-by-frame and
continuously merges observations into a persistent map, making it suitable for
real-time SLAM-style workflows.

```bash
# Simulate streaming from a directory of PCD files
tree-trunk-mapper stream scans/ --glob-pattern "*.pcd" --interval 0.1

# Skip every other frame for faster throughput
tree-trunk-mapper stream scans/ --process-every-n 2 -o live_map.json
```

Multi-frame accumulation improves accuracy: repeated observations of the same
trunk are merged, averaging out per-frame noise in position and DBH estimates.

### Key options

| Option | Default | Description |
|--------|---------|-------------|
| `--slice-height` | 1.3 | Height above ground for breast-height slice (m) |
| `--slice-thickness` | 0.2 | Thickness of the height slice (m) |
| `--eps` | 0.15 | DBSCAN epsilon (m) |
| `--min-samples` | 20 | DBSCAN minimum points per cluster |
| `--merge-radius` | 0.5 | Radius to merge detections across frames (m) |

## Example output

Detection on a FinnWoodlands scan with 5 trunks:

```
$ tree-trunk-mapper detect scan.pcd -o detections.json
Loading point cloud from scan.pcd ...
Detecting tree trunks ...
Found 5 trunk(s). Results saved to detections.json
```

```json
[
  {"center": [2.1534, 1.0782, 1.3012], "radius": 0.1247, "dbh": 0.2494, "inlier_count": 142},
  {"center": [4.5021, 3.2198, 1.2987], "radius": 0.0983, "dbh": 0.1966, "inlier_count": 118},
  {"center": [0.8812, 5.4401, 1.3045], "radius": 0.1521, "dbh": 0.3042, "inlier_count": 167},
  {"center": [6.2103, 2.8847, 1.2998], "radius": 0.0871, "dbh": 0.1742, "inlier_count": 95},
  {"center": [3.7290, 6.1125, 1.3021], "radius": 0.1105, "dbh": 0.2210, "inlier_count": 131}
]
```

## ROS2 Integration

The package includes an optional ROS2 node for real-time trunk detection from streaming LiDAR data with multi-frame accumulation.

### Launch

```bash
# With default parameters
ros2 launch tree_trunk_mapper trunk_detector.launch.py

# With custom topic and parameters
ros2 launch tree_trunk_mapper trunk_detector.launch.py \
    points_topic:=/velodyne_points \
    slice_height:=1.3 \
    eps:=0.15 \
    merge_radius:=0.5

# With parameter file
ros2 run tree_trunk_mapper trunk_detector_node \
    --ros-args --params-file config/default_params.yaml
```

### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/points` (input) | `sensor_msgs/PointCloud2` | Input LiDAR point cloud |
| `/trunk_map/markers` | `visualization_msgs/MarkerArray` | Green cylinders at detected trunk positions |
| `/trunk_map/poses` | `geometry_msgs/PoseArray` | Trunk positions (for downstream nodes) |
| `/trunk_map/geojson` | `std_msgs/String` | GeoJSON FeatureCollection of the trunk map |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `slice_height` | 1.3 | Height above ground for breast-height slice (m) |
| `slice_thickness` | 0.2 | Thickness of the height slice (m) |
| `eps` | 0.15 | DBSCAN epsilon (m) |
| `min_samples` | 20 | DBSCAN minimum points per cluster |
| `merge_radius` | 0.5 | Radius to merge detections across frames (m) |
| `min_points` | 100 | Skip frames with fewer points |

### rviz2 Visualization

1. Open rviz2
2. Add a **MarkerArray** display, set topic to `/trunk_map/markers`
3. Add a **PoseArray** display, set topic to `/trunk_map/poses`
4. Set the fixed frame to match your LiDAR frame (e.g., `velodyne`, `os_sensor`)

Detected trunks appear as green semi-transparent cylinders at their estimated positions.

## Testing

```bash
pytest -v
```

## License

MIT
