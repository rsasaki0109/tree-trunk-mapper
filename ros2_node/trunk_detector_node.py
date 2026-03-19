"""ROS2 node for real-time tree trunk detection from streaming LiDAR data."""

from __future__ import annotations

import json

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    from geometry_msgs.msg import PoseArray
    from visualization_msgs.msg import MarkerArray
    from std_msgs.msg import String
except ImportError as e:
    raise ImportError(
        "rclpy and ROS2 message packages are required to run the trunk detector node. "
        "Source your ROS2 workspace and install: rclpy, sensor-msgs, geometry-msgs, "
        "visualization-msgs, std-msgs."
    ) from e

import numpy as np

from tree_trunk_mapper.detector import detect_trunks
from tree_trunk_mapper.mapper import TrunkMapper
from ros2_node.conversions import pointcloud2_to_o3d, trunks_to_marker_array, trunks_to_pose_array


class TrunkDetectorNode(Node):
    """ROS2 node that detects tree trunks from streaming LiDAR point clouds.

    Subscribes to PointCloud2, runs the detection pipeline on each frame,
    accumulates detections via TrunkMapper, and publishes visualizations.
    """

    def __init__(self) -> None:
        super().__init__("trunk_detector_node")

        # Declare parameters with defaults
        self.declare_parameter("slice_height", 1.3)
        self.declare_parameter("slice_thickness", 0.2)
        self.declare_parameter("eps", 0.15)
        self.declare_parameter("min_samples", 20)
        self.declare_parameter("merge_radius", 0.5)
        self.declare_parameter("min_points", 100)

        # Read parameters
        self._slice_height = self.get_parameter("slice_height").value
        self._slice_thickness = self.get_parameter("slice_thickness").value
        self._eps = self.get_parameter("eps").value
        self._min_samples = self.get_parameter("min_samples").value
        merge_radius = self.get_parameter("merge_radius").value
        self._min_points = self.get_parameter("min_points").value

        # Trunk mapper for multi-frame accumulation
        self._mapper = TrunkMapper(merge_radius=merge_radius)

        # Subscriber
        self._sub = self.create_subscription(
            PointCloud2,
            "/points",
            self._pointcloud_callback,
            10,
        )

        # Publishers
        self._pub_markers = self.create_publisher(MarkerArray, "/trunk_map/markers", 10)
        self._pub_poses = self.create_publisher(PoseArray, "/trunk_map/poses", 10)
        self._pub_geojson = self.create_publisher(String, "/trunk_map/geojson", 10)

        self._frame_count = 0
        self.get_logger().info(
            f"TrunkDetectorNode started "
            f"(slice_height={self._slice_height}, eps={self._eps}, "
            f"merge_radius={merge_radius}, min_points={self._min_points})"
        )

    def _pointcloud_callback(self, msg: PointCloud2) -> None:
        """Process an incoming PointCloud2 message."""
        self._frame_count += 1

        # Convert to Open3D
        pcd = pointcloud2_to_o3d(msg)
        n_points = len(pcd.points)

        if n_points < self._min_points:
            self.get_logger().debug(
                f"Frame {self._frame_count}: skipping ({n_points} points < {self._min_points})"
            )
            return

        # Run detection
        detections = detect_trunks(
            pcd,
            slice_height=self._slice_height,
            slice_thickness=self._slice_thickness,
            eps=self._eps,
            min_samples=self._min_samples,
        )

        # Accumulate in mapper
        self._mapper.add_detections(detections)
        trunk_map = self._mapper.get_map()

        self.get_logger().info(
            f"Frame {self._frame_count}: {n_points} pts, "
            f"{len(detections)} new detections, {len(trunk_map)} total trunks"
        )

        frame_id = msg.header.frame_id
        stamp = msg.header.stamp

        # Publish MarkerArray (cylinders)
        marker_array = trunks_to_marker_array(trunk_map, frame_id, stamp)
        self._pub_markers.publish(marker_array)

        # Publish PoseArray
        pose_array = trunks_to_pose_array(trunk_map, frame_id, stamp)
        self._pub_poses.publish(pose_array)

        # Publish GeoJSON string
        geojson = self._build_geojson(trunk_map)
        geojson_msg = String()
        geojson_msg.data = json.dumps(geojson)
        self._pub_geojson.publish(geojson_msg)

    @staticmethod
    def _build_geojson(trunks) -> dict:
        """Build a GeoJSON FeatureCollection dict from trunk records."""
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
                    "dbh": round(trunk.dbh, 4),
                    "z": round(float(trunk.position[2]), 4),
                    "observation_count": trunk.observation_count,
                },
            }
            features.append(feature)
        return {"type": "FeatureCollection", "features": features}


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrunkDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
