from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_params = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            get_package_share_directory('pnavbot_nav2_bringup'),
            'config',
            'nav2_no_amcl.yaml'
        ),
        description='Full path to nav2 params yaml'
    )

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='False',
        description='Use simulation clock'
    )

    nav2_dir = get_package_share_directory('nav2_bringup')
    navigation_launch = os.path.join(nav2_dir, 'launch', 'navigation_launch.py')

    nav2_nav_only = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(navigation_launch),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': params_file
        }.items()
    )

    return LaunchDescription([
        declare_params,
        declare_use_sim_time,
        nav2_nav_only
    ])
