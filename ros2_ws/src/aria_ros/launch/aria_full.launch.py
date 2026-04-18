from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description() -> LaunchDescription:
    """Launch the full ARIA ROS2 node stack."""

    params = ["/opt/aria/configs/aria_params.yaml"]

    perception = LifecycleNode(
        package="aria_ros",
        executable="perception_node",
        name="aria_perception_node",
        namespace="aria",
        parameters=params,
        output="screen",
    )

    language = Node(
        package="aria_ros",
        executable="language_node",
        name="aria_language_node",
        namespace="aria",
        parameters=params,
        output="screen",
    )

    nav_policy = LifecycleNode(
        package="aria_ros",
        executable="nav_policy_node",
        name="aria_nav_policy_node",
        namespace="aria",
        parameters=params,
        output="screen",
    )

    manip_policy = Node(
        package="aria_ros",
        executable="manip_policy_node",
        name="aria_manip_policy_node",
        namespace="aria",
        parameters=params,
        output="screen",
    )

    return LaunchDescription([
        perception,
        language,
        nav_policy,
        manip_policy,
    ])
