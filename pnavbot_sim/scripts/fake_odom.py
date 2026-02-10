#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry

from tf2_ros import TransformBroadcaster


def yaw_to_quat(yaw: float):
    """Quaternion for yaw-only rotation."""
    # roll=pitch=0
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


class FakeOdom(Node):
    """
    Fake localization for simulation/dev:
    - subscribes /cmd_vel
    - integrates pose in 2D (x,y,yaw)
    - publishes /odom and TF odom->base_link
    """
    def __init__(self):
        super().__init__('fake_odom')

        # Parameters
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('publish_rate_hz', 50.0)

        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.odom_frame = self.get_parameter('odom_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        rate = max(1.0, rate)

        # State
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0

        self.last_time = self.get_clock().now()

        # ROS interfaces
        self.sub = self.create_subscription(Twist, self.cmd_vel_topic, self.cb_cmd, 10)
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(1.0 / rate, self.on_timer)

        self.get_logger().info(f"fake_odom started. Sub: {self.cmd_vel_topic}  Pub: {self.odom_topic}  TF: {self.odom_frame}->{self.base_frame}")

    def cb_cmd(self, msg: Twist):
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.wz = msg.angular.z

    def on_timer(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0.0:
            return

        # Integrate (robot frame velocities -> world frame)
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)

        # Convert body velocities to world
        vx_w = self.vx * cos_y - self.vy * sin_y
        vy_w = self.vx * sin_y + self.vy * cos_y

        self.x += vx_w * dt
        self.y += vy_w * dt
        self.yaw += self.wz * dt

        # normalize yaw
        self.yaw = (self.yaw + math.pi) % (2.0 * math.pi) - math.pi

        qx, qy, qz, qw = yaw_to_quat(self.yaw)

        # Publish TF odom -> base_link
        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = float(self.x)
        t.transform.translation.y = float(self.y)
        t.transform.translation.z = 0.0
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

        # Publish /odom
        odom = Odometry()
        odom.header.stamp = t.header.stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = float(self.x)
        odom.pose.pose.position.y = float(self.y)
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(self.vx)
        odom.twist.twist.linear.y = float(self.vy)
        odom.twist.twist.angular.z = float(self.wz)

        self.odom_pub.publish(odom)


def main():
    rclpy.init()
    node = FakeOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
