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

"""EasyNav implementation of the navigation backend."""

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TwistStamped

from .backend import NavigationBackend


class EasyNavNavigator(NavigationBackend):
    """Navigation backend that talks to EasyNav GoalManagerClient."""

    def __init__(self, node):
        super().__init__(node)
        try:
            from easynav_goalmanager_py import ClientState
            from easynav_goalmanager_py import GoalManagerClient
        except ImportError as e:
            raise RuntimeError(
                'navigation_backend=easynav requires '
                'easynav_goalmanager_py to be available in the sourced ROS '
                'environment'
            ) from e

        self._client_state = ClientState
        self._easynav_client = GoalManagerClient(self.node)
        self._first_pose_logged = False
        self._cmd_vel_pub = self.node.create_publisher(
            TwistStamped,
            'cmd_vel',
            10
        )

    @property
    def pose_topic(self) -> str:
        return 'localizer_node/simple/pose'

    def pose_qos(self):
        return 10

    def create_pose_subscription(self, callback):
        return self.node.create_subscription(
            PoseWithCovarianceStamped,
            self.pose_topic,
            callback,
            self.pose_qos()
        )

    def configure_initial_pose(self, initial_pose, frame, retry_period):
        self.node.get_logger().info(
            f'Waiting for first pose on [{self.pose_topic}]; EasyNav initial '
            'pose is configured through its parameter file'
        )

    def on_pose_received(self):
        if not self._first_pose_logged:
            self.node.get_logger().info(
                f'First localization pose received from [{self.pose_topic}]'
            )
            self._first_pose_logged = True

    def navigate(
        self,
        robot_name: str,
        pose,
        map_name: str,
        speed_limit=0.0
    ) -> bool:
        try:
            goal_msg = self._build_goal_pose(pose)
            self._easynav_client.send_goal(goal_msg)
            self._command_completed = False

            self.node.get_logger().info(
                f'[EASYNAV] Navigation goal sent to [{robot_name}]: {pose}'
            )
            return True

        except Exception as e:
            self.node.get_logger().error(f'EasyNav navigate() failed: {e}')
            self._command_completed = True
            return False

    def localize(self, robot_name: str, pose, map_name: str) -> bool:
        self.node.get_logger().warn(
            'localize() is not supported for EasyNav yet; configure the '
            'initial pose in the EasyNav parameter file'
        )
        return False

    def stop(self, robot_name: str) -> bool:
        try:
            state = self._easynav_client.get_state()
            if state == self._client_state.ACCEPTED_AND_NAVIGATING:
                self._easynav_client.cancel()
        except Exception as e:
            self.node.get_logger().error(f'EasyNav cancel failed: {e}')
            return False

        return self.publish_stop_velocity(self._cmd_vel_pub, robot_name)

    def position(self):
        return self.position_from_latest_pose()

    def is_command_completed(self) -> bool:
        state = self._easynav_client.get_state()
        terminal_states = (
            self._client_state.NAVIGATION_FINISHED,
            self._client_state.NAVIGATION_FAILED,
            self._client_state.NAVIGATION_CANCELLED,
            self._client_state.NAVIGATION_REJECTED,
            self._client_state.ERROR,
        )
        if state in terminal_states:
            self.node.get_logger().info(
                f'EasyNav command reached terminal state: {state.name}'
            )
            self._easynav_client.reset()
            self._command_completed = True
            return True

        return self._command_completed

    def _build_goal_pose(self, pose):
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_pose.header.frame_id = 'map'

        goal_pose.pose.position.x = float(pose[0])
        goal_pose.pose.position.y = float(pose[1])
        goal_pose.pose.position.z = 0.0

        qz, qw = self.quaternion_from_yaw(float(pose[2]))
        goal_pose.pose.orientation.z = qz
        goal_pose.pose.orientation.w = qw

        return goal_pose
