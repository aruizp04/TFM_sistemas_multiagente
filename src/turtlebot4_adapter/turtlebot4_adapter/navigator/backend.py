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

"""Common navigation backend contract and shared helpers."""

from abc import ABC
from abc import abstractmethod
import math
from typing import Optional

from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TwistStamped


class NavigationBackend(ABC):
    """Common contract for robot navigation backends."""

    def __init__(self, node):
        self.node = node
        self._pose = None
        self._localization_pose_received = False
        self._command_completed = True
        self._command_failed = False

    @property
    @abstractmethod
    def pose_topic(self) -> str:
        """Return the relative localization pose topic."""

    @abstractmethod
    def pose_qos(self):
        """Return the QoS setting used for the localization pose topic."""

    @abstractmethod
    def create_pose_subscription(self, callback):
        """Create the localization pose subscription."""

    @abstractmethod
    def configure_initial_pose(self, initial_pose, frame, retry_period):
        """Configure backend-specific initial pose behavior."""

    @abstractmethod
    def on_pose_received(self):
        """Handle backend-specific work after the first pose is received."""

    @abstractmethod
    def navigate(
        self,
        robot_name: str,
        pose,
        map_name: str,
        speed_limit=0.0
    ) -> bool:
        """Command the robot to navigate to a pose."""

    @abstractmethod
    def localize(self, robot_name: str, pose, map_name: str) -> bool:
        """Set or update the robot localization estimate."""

    @abstractmethod
    def stop(self, robot_name: str) -> bool:
        """Stop the robot's current navigation command."""

    @abstractmethod
    def position(self) -> Optional[list[float]]:
        """Return the latest robot pose as [x, y, yaw]."""

    @abstractmethod
    def is_command_completed(self) -> bool:
        """Return whether the current navigation command has completed."""

    def has_localized_pose(self) -> bool:
        """Return whether a real localization pose has been received."""
        return self._localization_pose_received

    def last_command_failed(self) -> bool:
        """Return whether the latest completed navigation command failed."""
        return self._command_failed

    @staticmethod
    def yaw_from_quaternion(q):
        """Convert a planar quaternion into yaw."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def quaternion_from_yaw(yaw: float):
        """Convert yaw into the z and w fields of a planar quaternion."""
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        return qz, qw

    def build_initial_pose_msg(self, pose, frame: str):
        """Build a PoseWithCovarianceStamped localization estimate."""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = frame

        msg.pose.pose.position.x = float(pose[0])
        msg.pose.pose.position.y = float(pose[1])
        msg.pose.pose.position.z = 0.0

        qz, qw = self.quaternion_from_yaw(float(pose[2]))
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.06853891945200942

        return msg

    def publish_stop_velocity(self, cmd_vel_pub, robot_name: str) -> bool:
        """Publish a zero velocity command."""
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

            cmd_vel_pub.publish(msg)
            self._command_completed = True

            self.node.get_logger().info(f'Stop command sent to [{robot_name}]')
            return True

        except Exception as e:
            self.node.get_logger().error(f'stop() failed: {e}')
            return False

    def store_pose(self, msg):
        """Store a localization pose message."""
        self._pose = msg.pose.pose
        self._localization_pose_received = True

    def position_from_latest_pose(self) -> Optional[list[float]]:
        """Return [x, y, yaw] from the latest pose."""
        if self._pose is None:
            return None

        x = self._pose.position.x
        y = self._pose.position.y
        theta = self.yaw_from_quaternion(self._pose.orientation)

        return [x, y, theta]
