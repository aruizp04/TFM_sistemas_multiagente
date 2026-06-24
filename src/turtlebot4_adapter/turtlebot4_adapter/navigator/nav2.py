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

"""Nav2 implementation of the navigation backend."""

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.clock import Clock
from rclpy.clock import ClockType
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

from .backend import NavigationBackend


class Nav2Navigator(NavigationBackend):
    """Navigation backend that talks to Nav2."""

    def __init__(self, node):
        super().__init__(node)
        self._goal_handle = None
        self._result_future = None
        self._first_pose_logged = False
        self._initial_pose = None
        self._initial_pose_frame = 'map'
        self._initial_pose_timer = None
        self._initial_pose_pub = self.node.create_publisher(
            PoseWithCovarianceStamped,
            'initialpose',
            10
        )
        self._cmd_vel_pub = self.node.create_publisher(
            self._twist_msg_type(),
            'cmd_vel',
            10
        )
        self._nav_to_pose_client = ActionClient(
            self.node,
            NavigateToPose,
            'navigate_to_pose'
        )

    @staticmethod
    def _twist_msg_type():
        from geometry_msgs.msg import TwistStamped
        return TwistStamped

    @property
    def pose_topic(self) -> str:
        return 'amcl_pose'

    def pose_qos(self):
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

    def create_pose_subscription(self, callback):
        return self.node.create_subscription(
            PoseWithCovarianceStamped,
            self.pose_topic,
            callback,
            self.pose_qos()
        )

    def configure_initial_pose(self, initial_pose, frame, retry_period):
        self._initial_pose = initial_pose
        self._initial_pose_frame = frame
        if initial_pose is not None:
            self._initial_pose_timer = self.node.create_timer(
                retry_period,
                self._publish_initial_pose_if_needed,
                clock=Clock(clock_type=ClockType.STEADY_TIME)
            )
            self.node.get_logger().warn(
                f'Waiting for first pose on [{self.pose_topic}]; automatic '
                'initial pose will be published every '
                f'{retry_period:.1f}s'
            )
        else:
            self.node.get_logger().warn(
                f'Waiting for first pose on [{self.pose_topic}]; no automatic '
                'initial_pose is configured'
            )

    def on_pose_received(self):
        if not self._first_pose_logged:
            self.node.get_logger().info(
                f'First localization pose received from [{self.pose_topic}]'
            )
            self._first_pose_logged = True
        if self._initial_pose_timer is not None:
            self._initial_pose_timer.cancel()

    def navigate(
        self,
        robot_name: str,
        pose,
        map_name: str,
        speed_limit=0.0
    ) -> bool:
        try:
            if not self._nav_to_pose_client.wait_for_server(timeout_sec=2.0):
                self.node.get_logger().error(
                    'Nav2 action server navigate_to_pose not available'
                )
                self._command_completed = True
                self._command_failed = True
                return False

            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = self._build_goal_pose(pose)

            self._command_completed = False
            self._command_failed = False
            self._goal_handle = None
            self._result_future = None

            send_goal_future = self._nav_to_pose_client.send_goal_async(
                goal_msg
            )
            send_goal_future.add_done_callback(self._goal_response_callback)

            self.node.get_logger().info(
                f'[NAV2] Navigation goal sent to [{robot_name}]: {pose}'
            )
            return True

        except Exception as e:
            self.node.get_logger().error(f'Nav2 navigate() failed: {e}')
            self._command_completed = True
            self._command_failed = True
            return False

    def localize(self, robot_name: str, pose, map_name: str) -> bool:
        try:
            msg = self.build_initial_pose_msg(pose, self._initial_pose_frame)
            self._initial_pose_pub.publish(msg)

            self.node.get_logger().info(
                f'Initial pose sent for [{robot_name}] on map [{map_name}]'
            )
            return True

        except Exception as e:
            self.node.get_logger().error(f'localize() failed: {e}')
            return False

    def stop(self, robot_name: str) -> bool:
        return self.publish_stop_velocity(self._cmd_vel_pub, robot_name)

    def position(self):
        return self.position_from_latest_pose()

    def is_command_completed(self) -> bool:
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

    def _publish_initial_pose_if_needed(self):
        if self._localization_pose_received or self._initial_pose is None:
            return

        try:
            msg = self.build_initial_pose_msg(
                self._initial_pose,
                self._initial_pose_frame
            )
            self._initial_pose_pub.publish(msg)
            self.node.get_logger().warn(
                'Publishing automatic initial pose for Nav2 localization: '
                f'{self._initial_pose} in frame [{self._initial_pose_frame}]'
            )
        except Exception as e:
            self.node.get_logger().error(
                f'Failed to publish automatic initial pose: {e}'
            )

    def _goal_response_callback(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.node.get_logger().warn('Nav2 goal rejected')
                self._command_completed = True
                self._command_failed = True
                return

            self.node.get_logger().info('Nav2 goal accepted')
            self._goal_handle = goal_handle
            self._result_future = goal_handle.get_result_async()
            self._result_future.add_done_callback(
                self._navigation_result_callback
            )

        except Exception as e:
            self.node.get_logger().error(f'Goal response callback failed: {e}')
            self._command_completed = True
            self._command_failed = True

    def _navigation_result_callback(self, future):
        try:
            result = future.result()
            self.node.get_logger().info(
                f'Nav2 goal finished with status: {result.status}'
            )
            self._command_failed = (
                result.status != GoalStatus.STATUS_SUCCEEDED
            )
        except Exception as e:
            self.node.get_logger().error(
                f'Navigation result callback failed: {e}'
            )
            self._command_failed = True

        self._command_completed = True
