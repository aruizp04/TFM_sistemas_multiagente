from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    namespace = LaunchConfiguration('namespace')
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value=''),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('params_file'),
        Node(
            package='my_world',
            executable='scan_stamp_delay.py',
            namespace=namespace,
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'input_topic': 'scan',
                'output_topic': 'scan_easynav',
                'delay_sec': 0.05,
            }],
            output='screen',
        ),
        Node(
            package='easynav_system',
            executable='system_main',
            namespace=namespace,
            parameters=[params_file],
            remappings=[
                ('cmd_vel_stamped', 'cmd_vel'),
                ('/tf', 'tf'),
                ('/tf_static', 'tf_static'),
            ],
            output='screen',
        ),
    ])
