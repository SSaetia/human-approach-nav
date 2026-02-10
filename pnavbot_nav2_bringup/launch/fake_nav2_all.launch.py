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
            rviz_node,
        ]
    )
