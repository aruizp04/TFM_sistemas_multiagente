# Copyright 2021 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import threading
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from sensor_msgs.msg import BatteryState

from .navigator import create_navigation_backend


class RobotAPI:
    """
    API minima para conectar RMF con un TurtleBot4 simulado.

    - Posicion Nav2: <namespace>/amcl_pose
    - Posicion EasyNav: <namespace>/localizer_node/simple/pose
    - Navegacion Nav2: <namespace>/navigate_to_pose
    - Navegacion EasyNav: <namespace>/easynav_control
    - Stop: /cmd_vel
    - Bateria: /battery_state
    """

    def __init__(self, config_yaml, use_sim_time=False, charger_name=None):
        self.prefix = config_yaml.get('prefix', '')
        self.user = config_yaml.get('user', '')
        self.password = config_yaml.get('password', '')
        self.namespace = config_yaml.get('namespace', '').strip('/')
        self.charger_name = charger_name
        self.initial_pose_source = config_yaml.get('initial_pose_source')
        self.initial_pose = self._resolve_initial_pose(config_yaml)
        self.initial_pose_frame = config_yaml.get('initial_pose_frame', 'map')
        self.initial_pose_retry_period = float(
            config_yaml.get('initial_pose_retry_period', 2.0)
        )
        self.report_initial_pose_until_localized = bool(
            config_yaml.get('report_initial_pose_until_localized', False)
        )
        self._reported_initial_pose_fallback = False

        self.timeout = 5.0
        self.debug = True

        self.map_name = 'L1'
        self.navigation_backend = config_yaml.get(
            'navigation_backend', 'nav2'
        ).lower()
        self.allow_navigation_before_localized = bool(
            config_yaml.get(
                'allow_navigation_before_localized',
                self.navigation_backend != 'nav2'
            )
        )

        self._last_missing_pose_warning_time = None
        self._battery_soc = 1.0

        if not rclpy.ok():
            rclpy.init(args=None)

        node_name = 'turtlebot4_robot_api'
        if self.namespace:
            node_name = f'{self.namespace}_robot_api'

        if self.namespace:
            self.node = rclpy.create_node(
                node_name,
                namespace=f'/{self.namespace}'
            )
        else:
            self.node = rclpy.create_node(node_name)

        if use_sim_time:
            self.node.set_parameters([
                Parameter('use_sim_time', Parameter.Type.BOOL, True)
            ])

        self.node.get_logger().info(
            f'Navigation backend selected: {self.navigation_backend}'
        )
        self.node.get_logger().info(
            f'Using robot namespace: /{self.namespace}' if self.namespace
            else 'Using global robot namespace'
        )

        self.navigator = create_navigation_backend(
            self.navigation_backend,
            self.node
        )

        self.pose_sub = self.navigator.create_pose_subscription(
            self._pose_callback
        )
        self.node.get_logger().info(
            f'Waiting for localization pose on [{self.navigator.pose_topic}]'
        )

        self.battery_sub = self.node.create_subscription(
            BatteryState,
            'battery_state',
            self._battery_callback,
            10
        )

        self.navigator.configure_initial_pose(
            self.initial_pose,
            self.initial_pose_frame,
            self.initial_pose_retry_period
        )

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        self.spin_thread = threading.Thread(
            target=self.executor.spin,
            daemon=True
        )
        self.spin_thread.start()

    def _resolve_initial_pose(self, config_yaml):
        if 'initial_pose' in config_yaml:
            return config_yaml.get('initial_pose')

        if config_yaml.get('initial_pose_source') != 'world_charger':
            return None

        world_package = config_yaml.get('initial_pose_world_package')
        world_name = config_yaml.get('initial_pose_world_name')
        initial_yaw = config_yaml.get('initial_pose_yaw')
        if not world_package or not world_name:
            raise ValueError(
                'initial_pose_source=world_charger requires '
                'initial_pose_world_package and initial_pose_world_name'
            )
        if self.charger_name is None:
            raise ValueError(
                'initial_pose_source=world_charger requires a robot charger'
            )
        if initial_yaw is None:
            raise ValueError(
                'initial_pose_source=world_charger requires initial_pose_yaw'
            )

        world_path = (
            Path(get_package_share_directory(world_package)) /
            'maps' / world_name / f'{world_name}.world'
        )
        try:
            root = ET.parse(world_path).getroot()
        except Exception as e:
            raise RuntimeError(
                f'Failed to parse initial pose world file [{world_path}]: {e}'
            ) from e

        for vertex in root.findall('.//rmf_charger_waypoints/rmf_vertex'):
            if vertex.get('name') == self.charger_name:
                return [
                    float(vertex.get('x')),
                    float(vertex.get('y')),
                    float(initial_yaw)
                ]

        raise RuntimeError(
            f'Charger [{self.charger_name}] was not found in world file '
            f'[{world_path}]'
        )

    def _pose_callback(self, msg):
        self.navigator.store_pose(msg)
        self.navigator.on_pose_received()

    def _battery_callback(self, msg: BatteryState):
        if msg.percentage >= 0.0:
            self._battery_soc = float(msg.percentage)

    def check_connection(self):
        """
        Devuelve True si el nodo ROS esta vivo.

        En esta primera version no exigimos haber recibido aun la pose,
        porque al arrancar puede tardar hasta que se haga 2D Pose Estimate.
        """
        return rclpy.ok()

    def localize(self, robot_name: str, pose, map_name: str):
        """Delegate localization to the active navigation backend."""
        localized = self.navigator.localize(robot_name, pose, map_name)
        if localized:
            self.map_name = map_name
        return localized

    def navigate(self, robot_name: str, pose, map_name: str, speed_limit=0.0):
        """Delegate navigation from RMF to the active backend."""
        if (
            self.navigation_backend == 'nav2' and
            not self.allow_navigation_before_localized and
            not self.has_localized_pose()
        ):
            self.node.get_logger().error(
                f'Refusing Nav2 navigation for [{robot_name}] before a real '
                f'pose is received on [{self.navigator.pose_topic}]'
            )
            return False

        return self.navigator.navigate(
            robot_name,
            pose,
            map_name,
            speed_limit
        )

    def start_activity(self, robot_name: str, activity: str, label: str):
        self.node.get_logger().info(
            f'Ignoring activity request [{activity}] with label [{label}]'
        )
        return True

    def stop(self, robot_name: str):
        """Stop through the active navigation backend."""
        return self.navigator.stop(robot_name)

    def position(self, robot_name: str):
        """
        Devuelve [x, y, theta] en el sistema de coordenadas del robot/Nav2.

        La transformacion a coordenadas RMF se define en config.yaml mediante
        reference_coordinates.
        """
        position = self.navigator.position()
        if position is not None:
            return position

        if (
            self.report_initial_pose_until_localized and
            self.initial_pose is not None
        ):
            if not self._reported_initial_pose_fallback:
                self.node.get_logger().warn(
                    f'Using configured initial pose for [{robot_name}] until '
                    f'[{self.navigator.pose_topic}] is received'
                )
                self._reported_initial_pose_fallback = True
            return [
                float(self.initial_pose[0]),
                float(self.initial_pose[1]),
                float(self.initial_pose[2])
            ]

        return None

    def battery_soc(self, robot_name: str):
        return self._battery_soc

    def map(self, robot_name: str):  # noqa: A003
        return self.map_name

    def is_command_completed(self):
        return self.navigator.is_command_completed()

    def has_localized_pose(self):
        return self.navigator.has_localized_pose()

    def last_command_failed(self):
        return self.navigator.last_command_failed()

    def get_data(self, robot_name: str):
        map_name = self.map(robot_name)
        position = self.position(robot_name)
        battery_soc = self.battery_soc(robot_name)

        if not (map_name is None or position is None or battery_soc is None):
            return RobotUpdateData(robot_name, map_name, position, battery_soc)

        if position is None:
            now = self.node.get_clock().now()
            warn = self._last_missing_pose_warning_time is None
            if not warn:
                elapsed = (
                    now - self._last_missing_pose_warning_time
                ).nanoseconds / 1e9
                warn = elapsed >= self.initial_pose_retry_period

            if warn:
                self.node.get_logger().warn(
                    'Waiting for localization pose on '
                    f'[{self.navigator.pose_topic}] before reporting '
                    f'[{robot_name}] to RMF'
                )
                self._last_missing_pose_warning_time = now

        return None


class RobotUpdateData:
    """Update data for a single robot."""

    def __init__(
        self,
        robot_name: str,
        map: str,  # noqa: A002
        position: list[float],
        battery_soc: float,
        requires_replan: bool | None = None
    ):
        self.robot_name = robot_name
        self.position = position
        self.map = map
        self.battery_soc = battery_soc
        self.requires_replan = requires_replan
