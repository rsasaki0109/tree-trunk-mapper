"""Convert between ROS2 messages and tree-trunk-mapper data structures."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    from tree_trunk_mapper.mapper import TrunkRecord

try:
    from geometry_msgs.msg import Pose, PoseArray
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import ColorRGBA, Header
    from visualization_msgs.msg import Marker, MarkerArray

    _HAS_ROS2 = True
except ImportError:
    _HAS_ROS2 = False


def _require_ros2() -> None:
    if not _HAS_ROS2:
        raise ImportError(
            "ROS2 message packages are required for conversions. "
            "Install rclpy, sensor-msgs, geometry-msgs, visualization-msgs."
        )


def pointcloud2_to_o3d(msg: PointCloud2) -> o3d.geometry.PointCloud:
    """Convert a sensor_msgs/PointCloud2 message to an Open3D PointCloud.

    Supports both structured (with named fields x, y, z) point clouds.
    Handles both float32 and float64 xyz fields.
    """
    _require_ros2()

    # Build a mapping of field name -> (offset, datatype)
    field_map: dict[str, tuple[int, int]] = {}
    for f in msg.fields:
        field_map[f.name] = (f.offset, f.datatype)

    if "x" not in field_map or "y" not in field_map or "z" not in field_map:
        raise ValueError("PointCloud2 message must contain x, y, z fields")

    # Determine format character for xyz based on datatype
    dtype_to_fmt = {
        PointField.FLOAT32: ("f", 4),
        PointField.FLOAT64: ("d", 8),
    }

    x_offset, x_dtype = field_map["x"]
    y_offset, y_dtype = field_map["y"]
    z_offset, z_dtype = field_map["z"]

    x_fmt, x_size = dtype_to_fmt.get(x_dtype, ("f", 4))
    y_fmt, y_size = dtype_to_fmt.get(y_dtype, ("f", 4))
    z_fmt, z_size = dtype_to_fmt.get(z_dtype, ("f", 4))

    data = bytes(msg.data)
    point_step = msg.point_step
    n_points = msg.width * msg.height

    points = np.empty((n_points, 3), dtype=np.float64)

    for i in range(n_points):
        base = i * point_step
        points[i, 0] = struct.unpack_from(x_fmt, data, base + x_offset)[0]
        points[i, 1] = struct.unpack_from(y_fmt, data, base + y_offset)[0]
        points[i, 2] = struct.unpack_from(z_fmt, data, base + z_offset)[0]

    # Filter out NaN / inf points
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def trunks_to_marker_array(
    trunks: list[TrunkRecord],
    frame_id: str,
    stamp,
) -> MarkerArray:
    """Convert TrunkRecord list to a visualization_msgs/MarkerArray of cylinders.

    Each trunk is shown as a green semi-transparent cylinder.
    """
    _require_ros2()

    marker_array = MarkerArray()

    # First, add a DELETE_ALL marker to clear old markers
    delete_marker = Marker()
    delete_marker.action = Marker.DELETEALL
    delete_marker.header.frame_id = frame_id
    delete_marker.header.stamp = stamp
    marker_array.markers.append(delete_marker)

    for trunk in trunks:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "tree_trunks"
        marker.id = trunk.trunk_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        # Position: trunk centre, shifted to ground level for visual
        marker.pose.position.x = float(trunk.position[0])
        marker.pose.position.y = float(trunk.position[1])
        marker.pose.position.z = float(trunk.position[2])
        marker.pose.orientation.w = 1.0

        # Scale: diameter for x/y, height for z
        marker.scale.x = float(trunk.dbh)
        marker.scale.y = float(trunk.dbh)
        marker.scale.z = 2.0  # 2m tall cylinder for visualization

        # Green semi-transparent color
        marker.color = ColorRGBA(r=0.2, g=0.8, b=0.2, a=0.7)

        marker.lifetime.sec = 0  # persistent

        marker_array.markers.append(marker)

    return marker_array


def trunks_to_pose_array(
    trunks: list[TrunkRecord],
    frame_id: str,
    stamp,
) -> PoseArray:
    """Convert TrunkRecord list to a geometry_msgs/PoseArray."""
    _require_ros2()

    pose_array = PoseArray()
    pose_array.header.frame_id = frame_id
    pose_array.header.stamp = stamp

    for trunk in trunks:
        pose = Pose()
        pose.position.x = float(trunk.position[0])
        pose.position.y = float(trunk.position[1])
        pose.position.z = float(trunk.position[2])
        pose.orientation.w = 1.0
        pose_array.poses.append(pose)

    return pose_array
