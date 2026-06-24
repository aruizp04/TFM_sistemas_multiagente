#!/usr/bin/env python3

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan


class ScanStampDelay(Node):
    def __init__(self):
        super().__init__('scan_stamp_delay')

        self.declare_parameter('input_topic', 'scan')
        self.declare_parameter('output_topic', 'scan_easynav')
        self.declare_parameter('delay_sec', 0.05)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.delay = Duration(
            seconds=float(self.get_parameter('delay_sec').value))

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE

        self.pub = self.create_publisher(LaserScan, output_topic, qos)
        self.sub = self.create_subscription(
            LaserScan, input_topic, self._on_scan, qos)

        self.get_logger().info(
            f'Republishing [{input_topic}] to [{output_topic}] '
            f'with stamp delay {self.delay.nanoseconds / 1e9:.3f}s')

    def _on_scan(self, msg):
        adjusted = msg

        stamp = Time.from_msg(msg.header.stamp)
        delayed_stamp = stamp - self.delay
        if delayed_stamp.nanoseconds < 0:
            delayed_stamp = Time(nanoseconds=0)

        adjusted.header.stamp = delayed_stamp.to_msg()
        self.pub.publish(adjusted)


def main():
    rclpy.init()
    node = ScanStampDelay()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
