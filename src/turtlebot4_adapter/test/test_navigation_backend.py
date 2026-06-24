# Copyright 2026 Open Source Robotics Foundation, Inc.
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

import builtins
from enum import Enum
from types import SimpleNamespace

from geometry_msgs.msg import PoseWithCovarianceStamped
import pytest


class FakeLogger:
    def info(self, message):
        self.last_info = message

    def warn(self, message):
        self.last_warn = message

    def error(self, message):
        self.last_error = message


class FakeClock:
    def now(self):
        return self

    def to_msg(self):
        return SimpleNamespace()


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class FakeTimer:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class FakeNode:
    def __init__(self):
        self.publishers = {}
        self.subscriptions = []
        self.timers = []
        self.logger = FakeLogger()
        self.parameters = []

    def create_publisher(self, msg_type, topic, qos):
        publisher = FakePublisher()
        self.publishers[topic] = publisher
        return publisher

    def create_subscription(self, msg_type, topic, callback, qos):
        subscription = (msg_type, topic, callback, qos)
        self.subscriptions.append(subscription)
        return subscription

    def create_timer(self, period, callback, clock=None):
        timer = FakeTimer()
        self.timers.append((period, callback, clock, timer))
        return timer

    def get_clock(self):
        return FakeClock()

    def get_logger(self):
        return self.logger

    def set_parameters(self, parameters):
        self.parameters.extend(parameters)


class FakeGoalFuture:
    def __init__(self, goal_handle):
        self.goal_handle = goal_handle
        self.callback = None

    def add_done_callback(self, callback):
        self.callback = callback

    def result(self):
        return self.goal_handle


class FakeActionClient:
    def __init__(self, node, action_type, topic):
        self.topic = topic
        self.goal = None
        self.goal_future = FakeGoalFuture(SimpleNamespace(accepted=True))

    def wait_for_server(self, timeout_sec):
        self.timeout_sec = timeout_sec
        return True

    def send_goal_async(self, goal_msg):
        self.goal = goal_msg
        return self.goal_future


def test_factory_validates_backend_and_keeps_easynav_lazy(monkeypatch):
    from turtlebot4_adapter.navigator import factory

    class FakeNav2:
        def __init__(self, node):
            self.node = node

    monkeypatch.setattr(factory, 'Nav2Navigator', FakeNav2)
    assert isinstance(factory.create_navigation_backend('nav2', object()), FakeNav2)

    with pytest.raises(ValueError):
        factory.create_navigation_backend('unknown', object())

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'easynav_goalmanager_py':
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    with pytest.raises(RuntimeError):
        factory.create_navigation_backend('easynav', FakeNode())


def test_nav2_navigator_builds_goal_and_publishes_initial_pose(monkeypatch):
    from turtlebot4_adapter.navigator import nav2

    fake_client = FakeActionClient(None, None, None)
    monkeypatch.setattr(nav2, 'ActionClient', lambda *args: fake_client)

    navigator = nav2.Nav2Navigator(FakeNode())

    assert navigator.navigate('robot1', [1.0, 2.0, 1.57], 'L1')
    goal_pose = fake_client.goal.pose
    assert goal_pose.header.frame_id == 'map'
    assert goal_pose.pose.position.x == 1.0
    assert goal_pose.pose.position.y == 2.0
    assert goal_pose.pose.orientation.w == pytest.approx(0.707388)
    assert navigator.is_command_completed() is False

    navigator.configure_initial_pose([3.0, 4.0, 0.0], 'map', 2.0)
    navigator._publish_initial_pose_if_needed()
    initial_pose = navigator._initial_pose_pub.messages[-1]
    assert initial_pose.pose.pose.position.x == 3.0
    assert initial_pose.pose.pose.position.y == 4.0

    rejected = FakeGoalFuture(SimpleNamespace(accepted=False))
    navigator._goal_response_callback(rejected)
    assert navigator.is_command_completed() is True
    assert navigator.last_command_failed() is True


def test_nav2_navigator_marks_failed_result(monkeypatch):
    from action_msgs.msg import GoalStatus
    from turtlebot4_adapter.navigator import nav2

    fake_client = FakeActionClient(None, None, None)
    monkeypatch.setattr(nav2, 'ActionClient', lambda *args: fake_client)

    navigator = nav2.Nav2Navigator(FakeNode())
    navigator.navigate('robot1', [1.0, 2.0, 0.0], 'L1')

    failed_future = SimpleNamespace(
        result=lambda: SimpleNamespace(status=GoalStatus.STATUS_ABORTED)
    )
    navigator._navigation_result_callback(failed_future)

    assert navigator.is_command_completed() is True
    assert navigator.last_command_failed() is True


def test_easynav_navigator_sends_goal_cancels_and_resets(monkeypatch):
    from turtlebot4_adapter.navigator import easynav

    class ClientState(Enum):
        ACCEPTED_AND_NAVIGATING = 1
        NAVIGATION_FINISHED = 2
        NAVIGATION_FAILED = 3
        NAVIGATION_CANCELLED = 4
        NAVIGATION_REJECTED = 5
        ERROR = 6

    class GoalManagerClient:
        last = None

        def __init__(self, node):
            self.goal = None
            self.cancelled = False
            self.reset_called = False
            self.state = ClientState.ACCEPTED_AND_NAVIGATING
            GoalManagerClient.last = self

        def send_goal(self, goal_msg):
            self.goal = goal_msg

        def get_state(self):
            return self.state

        def cancel(self):
            self.cancelled = True

        def reset(self):
            self.reset_called = True

    monkeypatch.setitem(
        __import__('sys').modules,
        'easynav_goalmanager_py',
        SimpleNamespace(
            ClientState=ClientState,
            GoalManagerClient=GoalManagerClient
        )
    )

    navigator = easynav.EasyNavNavigator(FakeNode())
    client = GoalManagerClient.last

    assert navigator.navigate('robot1', [1.0, 2.0, 0.0], 'L1')
    assert client.goal.header.frame_id == 'map'
    assert client.goal.pose.position.x == 1.0
    assert navigator.stop('robot1')
    assert client.cancelled is True

    client.state = ClientState.NAVIGATION_FINISHED
    assert navigator.is_command_completed() is True
    assert client.reset_called is True


def test_robot_api_delegates_to_navigator(monkeypatch):
    from turtlebot4_adapter import RobotClientAPI

    class FakeExecutor:
        def add_node(self, node):
            self.node = node

        def spin(self):
            return

    class FakeRclpy:
        def ok(self):
            return True

        def init(self, args=None):
            self.args = args

        def create_node(self, name, namespace=None):
            return FakeNode()

    class FakeNavigator:
        pose_topic = 'fake_pose'

        def __init__(self):
            self.localize_calls = []
            self.navigate_calls = []
            self.stop_calls = []
            self.position_value = [1.0, 2.0, 0.5]

        def create_pose_subscription(self, callback):
            self.pose_callback = callback
            return 'pose_sub'

        def configure_initial_pose(self, initial_pose, frame, retry_period):
            self.initial_pose_config = (initial_pose, frame, retry_period)

        def store_pose(self, msg):
            self.pose = msg

        def on_pose_received(self):
            self.pose_received = True

        def localize(self, robot_name, pose, map_name):
            self.localize_calls.append((robot_name, pose, map_name))
            return True

        def navigate(self, robot_name, pose, map_name, speed_limit=0.0):
            self.navigate_calls.append(
                (robot_name, pose, map_name, speed_limit)
            )
            return True

        def stop(self, robot_name):
            self.stop_calls.append(robot_name)
            return True

        def position(self):
            return self.position_value

        def is_command_completed(self):
            return True

        def has_localized_pose(self):
            return True

        def last_command_failed(self):
            return False

    fake_navigator = FakeNavigator()
    monkeypatch.setattr(RobotClientAPI, 'rclpy', FakeRclpy())
    monkeypatch.setattr(RobotClientAPI, 'SingleThreadedExecutor', FakeExecutor)
    monkeypatch.setattr(
        RobotClientAPI,
        'create_navigation_backend',
        lambda name, node: fake_navigator
    )

    api = RobotClientAPI.RobotAPI({'navigation_backend': 'nav2'})
    assert api.navigate('robot1', [1.0, 2.0, 0.0], 'L1', 0.2)
    assert fake_navigator.navigate_calls == [
        ('robot1', [1.0, 2.0, 0.0], 'L1', 0.2)
    ]

    assert api.localize('robot1', [0.0, 0.0, 0.0], 'L2')
    assert api.map('robot1') == 'L2'
    assert api.stop('robot1')

    api._battery_callback(SimpleNamespace(percentage=0.42))
    data = api.get_data('robot1')
    assert data.robot_name == 'robot1'
    assert data.map == 'L2'
    assert data.position == [1.0, 2.0, 0.5]
    assert data.battery_soc == 0.42

    msg = PoseWithCovarianceStamped()
    api._pose_callback(msg)
    assert fake_navigator.pose is msg
    assert fake_navigator.pose_received is True


def test_robot_api_can_report_initial_pose_until_localized(monkeypatch):
    from turtlebot4_adapter import RobotClientAPI

    class FakeExecutor:
        def add_node(self, node):
            self.node = node

        def spin(self):
            return

    class FakeRclpy:
        def ok(self):
            return True

        def init(self, args=None):
            self.args = args

        def create_node(self, name, namespace=None):
            return FakeNode()

    class FakeNavigator:
        pose_topic = 'amcl_pose'

        def create_pose_subscription(self, callback):
            return 'pose_sub'

        def configure_initial_pose(self, initial_pose, frame, retry_period):
            return

        def position(self):
            return None

        def localize(self, robot_name, pose, map_name):
            return False

        def navigate(self, robot_name, pose, map_name, speed_limit=0.0):
            return False

        def stop(self, robot_name):
            return True

        def is_command_completed(self):
            return True

        def has_localized_pose(self):
            return False

        def last_command_failed(self):
            return False

        def store_pose(self, msg):
            return

        def on_pose_received(self):
            return

    monkeypatch.setattr(RobotClientAPI, 'rclpy', FakeRclpy())
    monkeypatch.setattr(RobotClientAPI, 'SingleThreadedExecutor', FakeExecutor)
    monkeypatch.setattr(
        RobotClientAPI,
        'create_navigation_backend',
        lambda name, node: FakeNavigator()
    )

    api = RobotClientAPI.RobotAPI({
        'navigation_backend': 'nav2',
        'initial_pose': [3.0, 4.0, 1.5],
        'report_initial_pose_until_localized': True,
    })

    api._battery_callback(SimpleNamespace(percentage=0.5))
    data = api.get_data('robot1')

    assert data.robot_name == 'robot1'
    assert data.position == [3.0, 4.0, 1.5]
    assert data.battery_soc == 0.5


def test_robot_api_refuses_nav2_navigation_before_localized(monkeypatch):
    from turtlebot4_adapter import RobotClientAPI

    class FakeExecutor:
        def add_node(self, node):
            self.node = node

        def spin(self):
            return

    class FakeRclpy:
        def ok(self):
            return True

        def init(self, args=None):
            self.args = args

        def create_node(self, name, namespace=None):
            return FakeNode()

    class FakeNavigator:
        pose_topic = 'amcl_pose'

        def __init__(self):
            self.navigate_calls = []

        def create_pose_subscription(self, callback):
            return 'pose_sub'

        def configure_initial_pose(self, initial_pose, frame, retry_period):
            return

        def navigate(self, robot_name, pose, map_name, speed_limit=0.0):
            self.navigate_calls.append(
                (robot_name, pose, map_name, speed_limit)
            )
            return True

        def has_localized_pose(self):
            return False

        def last_command_failed(self):
            return False

        def is_command_completed(self):
            return True

        def position(self):
            return None

        def localize(self, robot_name, pose, map_name):
            return False

        def stop(self, robot_name):
            return True

        def store_pose(self, msg):
            return

        def on_pose_received(self):
            return

    fake_navigator = FakeNavigator()
    monkeypatch.setattr(RobotClientAPI, 'rclpy', FakeRclpy())
    monkeypatch.setattr(RobotClientAPI, 'SingleThreadedExecutor', FakeExecutor)
    monkeypatch.setattr(
        RobotClientAPI,
        'create_navigation_backend',
        lambda name, node: fake_navigator
    )

    api = RobotClientAPI.RobotAPI({'navigation_backend': 'nav2'})

    assert api.navigate('robot1', [1.0, 2.0, 0.0], 'L1') is False
    assert fake_navigator.navigate_calls == []


def test_robot_adapter_does_not_keep_execution_after_rejected_command():
    from turtlebot4_adapter.fleet_adapter import RobotAdapter

    class FakeAPI:
        def __init__(self):
            self.navigate_calls = []

        def navigate(self, robot_name, pose, map_name, speed_limit=0.0):
            self.navigate_calls.append(
                (robot_name, pose, map_name, speed_limit)
            )
            return False

    class FakeMore:
        def __init__(self):
            self.replan_calls = 0

        def replan(self):
            self.replan_calls += 1

    class FakeUpdateHandle:
        def __init__(self):
            self.more_handle = FakeMore()

        def more(self):
            return self.more_handle

    node = FakeNode()
    api = FakeAPI()
    adapter = RobotAdapter('robot1', object(), node, api, object())
    adapter.update_handle = FakeUpdateHandle()
    execution = SimpleNamespace(identifier=object())
    destination = SimpleNamespace(
        position=[1.0, 2.0, 0.0],
        map='L1',
        speed_limit=0.2
    )

    adapter.navigate(destination, execution)

    assert adapter.execution is None
    assert api.navigate_calls == [('robot1', [1.0, 2.0, 0.0], 'L1', 0.2)]
    assert adapter.update_handle.more_handle.replan_calls == 1


def test_robot_adapter_replans_after_async_navigation_failure():
    from turtlebot4_adapter.fleet_adapter import RobotAdapter

    class FakeAPI:
        def is_command_completed(self):
            return True

        def last_command_failed(self):
            return True

    class FakeMore:
        def __init__(self):
            self.replan_calls = 0

        def replan(self):
            self.replan_calls += 1

    class FakeUpdateHandle:
        def __init__(self):
            self.more_handle = FakeMore()
            self.update_calls = []

        def more(self):
            return self.more_handle

        def update(self, state, activity_identifier):
            self.update_calls.append((state, activity_identifier))

    execution = SimpleNamespace(
        identifier=SimpleNamespace(is_same=lambda activity: True),
        finished=lambda: pytest.fail('failed navigation must not finish')
    )
    adapter = RobotAdapter('robot1', object(), FakeNode(), FakeAPI(), object())
    adapter.execution = execution
    adapter.update_handle = FakeUpdateHandle()

    adapter.update(SimpleNamespace())

    assert adapter.execution is None
    assert adapter.update_handle.more_handle.replan_calls == 1
