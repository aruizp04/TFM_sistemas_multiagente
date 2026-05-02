# Copyright 2021 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0

import math
import threading
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TwistStamped
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import BatteryState


class RobotAPI:
    """
    API minima para conectar RMF con un TurtleBot4 simulado.

    - Posicion: /amcl_pose
    - Navegacion Nav2: /navigate_to_pose
    - Navegacion EasyNav: pendiente de implementar
    - Stop: /cmd_vel
    - Bateria: /battery_state
    """

    def __init__(self, config_yaml):
        self.prefix = config_yaml.get('prefix', '')
        self.user = config_yaml.get('user', '')
        self.password = config_yaml.get('password', '')

        self.timeout = 5.0
        self.debug = True

        self.map_name = 'L1'

        # Selector de backend de navegacion.
        # Valores esperados:
        #   - nav2
        #   - easynav
        self.navigation_backend = config_yaml.get('navigation_backend', 'nav2')

        self._pose = None
        self._battery_soc = 1.0

        self._goal_handle = None
        self._result_future = None
        self._command_completed = True

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = rclpy.create_node('turtlebot4_robot_api')

        self.node.get_logger().info(
            f'Navigation backend selected: {self.navigation_backend}'
        )

        self.pose_sub = self.node.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self._amcl_pose_callback,
            10
        )

        self.battery_sub = self.node.create_subscription(
            BatteryState,
            '/battery_state',
            self._battery_callback,
            10
        )

        self.cmd_vel_pub = self.node.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        # Cliente de Nav2.
        # Se mantiene igual que en tu implementacion actual.
        self.nav_to_pose_client = None
        if self.navigation_backend == 'nav2':
            self.nav_to_pose_client = ActionClient(
                self.node,
                NavigateToPose,
                '/navigate_to_pose'
            )

        # Aqui se podran declarar publishers, services o action clients de EasyNav
        # cuando se decida cual es su interfaz real.
        self.easynav_client = None

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        self.spin_thread = threading.Thread(
            target=self.executor.spin,
            daemon=True
        )
        self.spin_thread.start()

    def _amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        self._pose = msg.pose.pose

    def _battery_callback(self, msg: BatteryState):
        if msg.percentage >= 0.0:
            self._battery_soc = float(msg.percentage)

    @staticmethod
    def _yaw_from_quaternion(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _quaternion_from_yaw(yaw: float):
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        return qz, qw

    def check_connection(self):
        """
        Devuelve True si el nodo ROS esta vivo.

        En esta primera version no exigimos haber recibido aun /amcl_pose,
        porque al arrancar puede tardar hasta que se haga 2D Pose Estimate.
        """
        return rclpy.ok()

    def localize(self, robot_name: str, pose, map_name: str):
        """
        Publica una estimacion inicial en /initialpose.

        pose llega como [x, y, theta].
        """
        try:
            pub = self.node.create_publisher(
                PoseWithCovarianceStamped,
                '/initialpose',
                10
            )

            msg = PoseWithCovarianceStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = 'map'

            msg.pose.pose.position.x = float(pose[0])
            msg.pose.pose.position.y = float(pose[1])
            msg.pose.pose.position.z = 0.0

            qz, qw = self._quaternion_from_yaw(float(pose[2]))
            msg.pose.pose.orientation.z = qz
            msg.pose.pose.orientation.w = qw

            pub.publish(msg)
            self.map_name = map_name

            self.node.get_logger().info(
                f'Initial pose sent for [{robot_name}] on map [{map_name}]'
            )
            return True

        except Exception as e:
            self.node.get_logger().error(f'localize() failed: {e}')
            return False

    def navigate(self, robot_name: str, pose, map_name: str, speed_limit=0.0):
        """
        Punto unico de entrada para navegacion desde RMF.

        Dependiendo de navigation_backend, redirige la orden a:
        - Nav2
        - EasyNav
        """
        if self.navigation_backend == 'nav2':
            return self._navigate_with_nav2(
                robot_name,
                pose,
                map_name,
                speed_limit
            )

        if self.navigation_backend == 'easynav':
            return self._navigate_with_easynav(
                robot_name,
                pose,
                map_name,
                speed_limit
            )

        self.node.get_logger().error(
            f'Unknown navigation backend: {self.navigation_backend}'
        )
        return False

    def _navigate_with_nav2(
        self,
        robot_name: str,
        pose,
        map_name: str,
        speed_limit=0.0
    ):
        """
        Nav2.

        Se mantiene igual que tu navigate() original:
        - Usa /navigate_to_pose
        - Envia NavigateToPose.Goal
        - Usa callbacks de goal/result
        """
        try:
            if self.nav_to_pose_client is None:
                self.node.get_logger().error(
                    'Nav2 backend selected but nav_to_pose_client is not initialized'
                )
                return False

            if not self.nav_to_pose_client.wait_for_server(timeout_sec=2.0):
                self.node.get_logger().error(
                    'Nav2 action server /navigate_to_pose not available'
                )
                return False

            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = PoseStamped()
            goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
            goal_msg.pose.header.frame_id = 'map'

            goal_msg.pose.pose.position.x = float(pose[0])
            goal_msg.pose.pose.position.y = float(pose[1])
            goal_msg.pose.pose.position.z = 0.0

            qz, qw = self._quaternion_from_yaw(float(pose[2]))
            goal_msg.pose.pose.orientation.z = qz
            goal_msg.pose.pose.orientation.w = qw

            self._command_completed = False
            self._goal_handle = None
            self._result_future = None

            send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
            send_goal_future.add_done_callback(self._goal_response_callback)

            self.node.get_logger().info(
                f'[NAV2] Navigation goal sent to [{robot_name}]: {pose}'
            )
            return True

        except Exception as e:
            self.node.get_logger().error(f'_navigate_with_nav2() failed: {e}')
            self._command_completed = True
            return False

    def _navigate_with_easynav(
        self,
        robot_name: str,
        pose,
        map_name: str,
        speed_limit=0.0
    ):
        """
        EASYNAV
        """
        self.node.get_logger().warn(
            f'[EASYNAV] Navigation requested for [{robot_name}] '
            f'to pose {pose} on map [{map_name}], but EasyNav is not implemented yet'
        )

        self._command_completed = True
        return False

    def _goal_response_callback(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.node.get_logger().warn('Nav2 goal rejected')
                self._command_completed = True
                return

            self.node.get_logger().info('Nav2 goal accepted')
            self._goal_handle = goal_handle
            self._result_future = goal_handle.get_result_async()
            self._result_future.add_done_callback(self._navigation_result_callback)

        except Exception as e:
            self.node.get_logger().error(f'Goal response callback failed: {e}')
            self._command_completed = True

    def _navigation_result_callback(self, future):
        try:
            result = future.result()
            self.node.get_logger().info(
                f'Nav2 goal finished with status: {result.status}'
            )
        except Exception as e:
            self.node.get_logger().error(f'Navigation result callback failed: {e}')

        self._command_completed = True

    def start_activity(self, robot_name: str, activity: str, label: str):
        self.node.get_logger().info(
            f'Ignoring activity request [{activity}] with label [{label}]'
        )
        return True

    def stop(self, robot_name: str):
        """
        Publica velocidad cero.

        De momento se mantiene comun para Nav2 y EasyNav.
        """
        try:
            msg = TwistStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'

            msg.twist.linear.x = 0.0
            msg.twist.linear.y = 0.0
            msg.twist.linear.z = 0.0
            msg.twist.angular.x = 0.0
            msg.twist.angular.y = 0.0
            msg.twist.angular.z = 0.0

            self.cmd_vel_pub.publish(msg)
            self._command_completed = True

            self.node.get_logger().info(f'Stop command sent to [{robot_name}]')
            return True

        except Exception as e:
            self.node.get_logger().error(f'stop() failed: {e}')
            return False

    def position(self, robot_name: str):
        """
        Devuelve [x, y, theta] en el sistema de coordenadas del robot/Nav2.

        La transformacion a coordenadas RMF se define en config.yaml mediante
        reference_coordinates.
        """
        if self._pose is None:
            return None

        x = self._pose.position.x
        y = self._pose.position.y
        theta = self._yaw_from_quaternion(self._pose.orientation)

        return [x, y, theta]

    def battery_soc(self, robot_name: str):
        return self._battery_soc

    def map(self, robot_name: str):
        return self.map_name

    def is_command_completed(self):
        return self._command_completed

    def get_data(self, robot_name: str):
        map_name = self.map(robot_name)
        position = self.position(robot_name)
        battery_soc = self.battery_soc(robot_name)

        if not (map_name is None or position is None or battery_soc is None):
            return RobotUpdateData(robot_name, map_name, position, battery_soc)

        return None


class RobotUpdateData:
    """Update data for a single robot."""

    def __init__(
        self,
        robot_name: str,
        map: str,
        position: list[float],
        battery_soc: float,
        requires_replan: bool | None = None
    ):
        self.robot_name = robot_name
        self.position = position
        self.map = map
        self.battery_soc = battery_soc
        self.requires_replan = requires_replan