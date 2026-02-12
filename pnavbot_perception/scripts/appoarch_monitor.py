#!/usr/bin/env python3
import os
import csv
import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav2_msgs.msg import SpeedLimit

import tf2_ros
from tf2_ros import TransformException


def yaw_from_quat(x, y, z, w) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class ApproachMonitor(Node):
    """
    Collect evaluation data for human-aware approach:
    - /human/pose (PoseStamped)
    - /approach/goal_pose (PoseStamped)  [published by approach_goal_sender]
    - /speed_limit (nav2_msgs/SpeedLimit)
    - robot pose from TF: map -> base_link
    Write rows to CSV at a fixed rate.
    """

    def __init__(self):
        super().__init__("approach_monitor")

        # -------- Parameters --------
        self.declare_parameter("human_topic", "/human/pose")
        self.declare_parameter("goal_topic", "/approach/goal_pose")
        self.declare_parameter("speed_topic", "/speed_limit")

        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_frame", "base_link")

        self.declare_parameter("rate_hz", 10.0)

        # CSV output
        self.declare_parameter(
            "output_dir",
            "/home/hayashi/pat_ws/src/pnavbot/pnavbot_perception/data"
        )
        self.declare_parameter("filename", "approach_monitor_log.csv")
        self.declare_parameter("flush_every_n", 20)

        # -------- Load params --------
        self.human_topic = self.get_parameter("human_topic").value
        self.goal_topic = self.get_parameter("goal_topic").value
        self.speed_topic = self.get_parameter("speed_topic").value

        self.global_frame = self.get_parameter("global_frame").value
        self.robot_frame = self.get_parameter("robot_frame").value

        self.rate_hz = float(self.get_parameter("rate_hz").value)

        self.output_dir = self.get_parameter("output_dir").value
        self.filename = self.get_parameter("filename").value
        self.flush_every_n = int(self.get_parameter("flush_every_n").value)

        # -------- Subscribers --------
        self.last_human: Optional[PoseStamped] = None
        self.last_goal: Optional[PoseStamped] = None
        self.last_speed: Optional[SpeedLimit] = None

        self.create_subscription(PoseStamped, self.human_topic, self._on_human, 10)
        self.create_subscription(PoseStamped, self.goal_topic, self._on_goal, 10)
        self.create_subscription(SpeedLimit, self.speed_topic, self._on_speed, 10)

        # -------- TF --------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # -------- CSV setup --------
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, self.filename)
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            # time
            "t_sec", "t_nanosec",

            # human pose
            "human_x", "human_y", "human_yaw_rad", "human_yaw_deg",

            # goal pose
            "goal_x", "goal_y", "goal_yaw_rad", "goal_yaw_deg",

            # robot pose
            "robot_x", "robot_y", "robot_yaw_rad", "robot_yaw_deg",

            # distances
            "dist_robot_human",
            "dist_robot_goal",
            "dist_goal_human",

            # speed limit
            "speed_percentage",
            "speed_limit_value",

            # misc
            "human_frame",
            "goal_frame",
            "robot_frame",
        ])

        self.row_count = 0
        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)

        self.get_logger().info(f"ApproachMonitor logging to: {self.csv_path}")
        self.get_logger().info(f"Subscribing: human={self.human_topic}, goal={self.goal_topic}, speed={self.speed_topic}")

    def _on_human(self, msg: PoseStamped):
        self.last_human = msg

    def _on_goal(self, msg: PoseStamped):
        self.last_goal = msg

    def _on_speed(self, msg: SpeedLimit):
        self.last_speed = msg

    def _get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time()
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
            return (t.x, t.y, yaw)
        except TransformException as ex:
            self.get_logger().warn(f"TF not ready: {ex}", throttle_duration_sec=2.0)
            return None

    def _pose_xyyaw(self, msg: PoseStamped) -> Tuple[float, float, float]:
        x = msg.pose.position.x
        y = msg.pose.position.y
        q = msg.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        return (x, y, yaw)

    def _tick(self):
        now = self.get_clock().now().to_msg()

        # We can still log partial rows (use NaN) if something missing
        human_x = human_y = human_yaw = float("nan")
        goal_x = goal_y = goal_yaw = float("nan")
        robot_x = robot_y = robot_yaw = float("nan")

        human_frame = ""
        goal_frame = ""

        if self.last_human is not None:
            human_x, human_y, human_yaw = self._pose_xyyaw(self.last_human)
            human_frame = self.last_human.header.frame_id

        if self.last_goal is not None:
            goal_x, goal_y, goal_yaw = self._pose_xyyaw(self.last_goal)
            goal_frame = self.last_goal.header.frame_id

        robot_pose = self._get_robot_pose()
        if robot_pose is not None:
            robot_x, robot_y, robot_yaw = robot_pose

        # distances (only if values are valid)
        def safe_dist(ax, ay, bx, by):
            if any(math.isnan(v) for v in [ax, ay, bx, by]):
                return float("nan")
            return math.hypot(ax - bx, ay - by)

        dist_rh = safe_dist(robot_x, robot_y, human_x, human_y)
        dist_rg = safe_dist(robot_x, robot_y, goal_x, goal_y)
        dist_gh = safe_dist(goal_x, goal_y, human_x, human_y)

        # speed limit
        speed_percentage = ""
        speed_limit_value = float("nan")
        if self.last_speed is not None:
            speed_percentage = str(bool(self.last_speed.percentage))
            speed_limit_value = float(self.last_speed.speed_limit)

        # write row
        self.writer.writerow([
            now.sec, now.nanosec,

            human_x, human_y, human_yaw, math.degrees(human_yaw) if not math.isnan(human_yaw) else float("nan"),
            goal_x, goal_y, goal_yaw, math.degrees(goal_yaw) if not math.isnan(goal_yaw) else float("nan"),
            robot_x, robot_y, robot_yaw, math.degrees(robot_yaw) if not math.isnan(robot_yaw) else float("nan"),

            dist_rh, dist_rg, dist_gh,

            speed_percentage,
            speed_limit_value,

            human_frame,
            goal_frame,
            self.robot_frame,
        ])

        self.row_count += 1
        if self.row_count % self.flush_every_n == 0:
            self.csv_file.flush()

    def destroy_node(self):
        try:
            if hasattr(self, "csv_file") and self.csv_file:
                self.csv_file.flush()
                self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = ApproachMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
