from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # ---- Args ----
    map_yaml = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')

    start_x = LaunchConfiguration('start_x')
    start_y = LaunchConfiguration('start_y')
    start_yaw = LaunchConfiguration('start_yaw')

    rviz_config = LaunchConfiguration('rviz_config')
    use_rviz = LaunchConfiguration('use_rviz')

    nav2_pkg = get_package_share_directory('pnavbot_nav2_bringup')
    sim_pkg = get_package_share_directory('pnavbot_sim')

    default_map = os.path.join(sim_pkg, 'maps', 'anna_map.yaml')
    default_rviz = os.path.join(nav2_pkg, 'config', 'navigation.rviz')

    declare_args = [
        DeclareLaunchArgument(
            'map',
            default_value=default_map,
            description='Full path to map yaml'
        ),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('start_x', default_value='3.2'),
        DeclareLaunchArgument('start_y', default_value='-1.7'),
        DeclareLaunchArgument('start_yaw', default_value='0.0'),

        # RViz
        DeclareLaunchArgument(
            'rviz_config',
            default_value=default_rviz,
            description='Full path to RViz config file'
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Start RViz2'
        ),

        # ---- Human replay params (pose publisher) ----
        DeclareLaunchArgument('human_x_offset', default_value='6.0'),
        DeclareLaunchArgument('human_y_offset', default_value='-5.5'),
        DeclareLaunchArgument('human_yaw_noise_std', default_value='0.2'),
        DeclareLaunchArgument('human_yaw_smoothing', default_value='0.2'),

    ]

    # ---- 1) Map server ----
    map_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sim_pkg, 'launch', 'anna_map_server.launch.py')
        ),
        launch_arguments={
            'map': map_yaml,
            'use_sim_time': use_sim_time,
        }.items()
    )

    # ---- 2) Static TF: map -> odom (sets spawn pose in map) ----
    static_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_to_odom',
        arguments=[start_x, start_y, '0', start_yaw, '0', '0', 'map', 'odom'],
        output='screen'
    )

    # ---- 3) Fake odom ----
    fake_odom = Node(
        package='pnavbot_sim',
        executable='fake_odom.py',
        name='fake_odom',
        output='screen',
        parameters=[
            {'cmd_vel_topic': '/cmd_vel'},
            {'use_sim_time': use_sim_time},
        ],
    )

    # ---- 4) Nav2 bringup (no AMCL) ----
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_pkg, 'launch', 'nav2_no_amcl.launch.py')
        ),
        launch_arguments={
            'map': map_yaml,
            'use_sim_time': use_sim_time,
            'localization': 'false',
            'slam': 'false',
        }.items()
    )

    # ---- Human (pose replay only) ----
    human_replay = Node(
        package='pnavbot_sim',
        executable='human_replay.py',
        name='human_replay',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'x_offset': LaunchConfiguration('human_x_offset'),
            'y_offset': LaunchConfiguration('human_y_offset'),
            'yaw_noise_std': LaunchConfiguration('human_yaw_noise_std'),
            'yaw_smoothing': LaunchConfiguration('human_yaw_smoothing'),
            # keep publishing pose in map
            'map_frame': 'map',
        }],
    )

    # ---- Human obstacle cloud generator (real-world-style) ----
    human_obstacle_cloud = Node(
        package='pnavbot_perception',
        executable='human_obstacle_cloud.py',   # you will create this
        name='human_obstacle_cloud',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'input_pose_topic': '/human/pose',
            'output_cloud_topic': '/human/obstacles',
            'odom_frame': 'odom',
            'ring_radius_m': 1.0,
            'ring_points': 36,
            'ring_z': 0.0,
        }],
    )


    # ---- 5) RViz ----
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        # Simple on/off switch:
        condition=None  # keep always on (see note below)
    )

    # NOTE: launch conditions require "from launch.conditions import IfCondition"
    # If you want use_rviz:=false support, I’ll give you the 2-line add.

    return LaunchDescription(
        declare_args + [
            map_server_launch,
            static_map_odom,
            fake_odom,
            nav2_launch,
            human_replay,
            human_obstacle_cloud,
            rviz_node,
        ]
    )
