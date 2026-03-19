"""Launch file for the trunk_detector_node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        # Declare launch arguments for all tunable parameters
        DeclareLaunchArgument("points_topic", default_value="/points",
                              description="Input PointCloud2 topic"),
        DeclareLaunchArgument("slice_height", default_value="1.3",
                              description="Height above ground for breast-height slice (m)"),
        DeclareLaunchArgument("slice_thickness", default_value="0.2",
                              description="Thickness of the height slice (m)"),
        DeclareLaunchArgument("eps", default_value="0.15",
                              description="DBSCAN epsilon (m)"),
        DeclareLaunchArgument("min_samples", default_value="20",
                              description="DBSCAN minimum points per cluster"),
        DeclareLaunchArgument("merge_radius", default_value="0.5",
                              description="Radius to merge detections across frames (m)"),
        DeclareLaunchArgument("min_points", default_value="100",
                              description="Skip frames with fewer points"),

        Node(
            package="tree_trunk_mapper",
            executable="trunk_detector_node",
            name="trunk_detector_node",
            output="screen",
            parameters=[{
                "slice_height": LaunchConfiguration("slice_height"),
                "slice_thickness": LaunchConfiguration("slice_thickness"),
                "eps": LaunchConfiguration("eps"),
                "min_samples": LaunchConfiguration("min_samples"),
                "merge_radius": LaunchConfiguration("merge_radius"),
                "min_points": LaunchConfiguration("min_points"),
            }],
            remappings=[
                ("/points", LaunchConfiguration("points_topic")),
            ],
        ),
    ])
