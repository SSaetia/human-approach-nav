#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
approach_goal_sender.py (Nav2 Humble)

Clean version:
- Sub /human/pose (map)
- TF map->base_link
- Social goal selection (front-right sector) + optional staging if robot is behind
- Path gate using /compute_path_to_pose:
    Reject if:
      (A) path goes behind human within behind_guard_radius
      (B) path enters personal_space_radius (too close)
- Safety:
    STOP if robot too close to human (stop_radius): cancel nav + publish near-zero SpeedLimit
    SLOW if within slowdown_radius
- Anti-churn:
    Only replan if:
      (1) human moved/turned enough OR
      (2) we have no active goal OR
      (3) last goal became invalid (aborted) OR
      (4) min_goal_period passed and goal changed enough

Status mapping (ROS2 action_msgs/GoalStatus):
  4 = SUCCEEDED
  5 = CANCELED
  6 = ABORTED
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose, ComputePathToPose
from nav2_msgs.msg import SpeedLimit

import tf2_ros
from tf2_ros import TransformException


# -------------------- Helpers --------------------
def wrap_to_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def dist2d(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


# -------------------- Data --------------------
@dataclass
class HumanState:
    x: float
    y: float
    yaw: float
    stamp_sec: int
    stamp_nanosec: int


# -------------------- Node --------------------
class ApproachGoalSender(Node):
    def __init__(self):
        super().__init__('approach_goal_sender')

        # ========== Parameters ==========
        self.declare_parameter('human_topic', '/human/pose')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')

        self.declare_parameter('nav_action_name', '/navigate_to_pose')
        self.declare_parameter('path_action_name', '/compute_path_to_pose')

        # Final approach: front-right of human
        self.declare_parameter('final_distance', 1.0)              # meters
        self.declare_parameter('candidate_angles_deg', [-60.0, -45.0, -30.0, -75.0, -90.0])

        # Allowed sector (relative to human facing): 0=front, -90=right
        self.declare_parameter('allowed_min_deg', -90.0)
        self.declare_parameter('allowed_max_deg', -15.0)

        # Behind guard + staging
        self.declare_parameter('behind_guard_radius', 3.0)         # meters
        self.declare_parameter('staging_distance', 2.2)            # meters
        self.declare_parameter('staging_angle_deg', -90.0)         # right side

        # Path gates
        self.declare_parameter('path_gate_step', 0.20)             # meters
        self.declare_parameter('personal_space_radius', 1.0)       # meters (reject path too close)

        # Anti-churn
        self.declare_parameter('min_goal_period', 1.2)             # seconds
        self.declare_parameter('goal_update_dist_thresh', 0.40)    # meters
        self.declare_parameter('goal_update_yaw_thresh_deg', 20.0) # degrees

        # HUMAN-change trigger (fix: do not stop forever after success)
        self.declare_parameter('human_update_dist_thresh', 0.25)   # meters
        self.declare_parameter('human_update_yaw_thresh_deg', 12.0)# degrees

        self.declare_parameter('control_rate_hz', 10.0)

        # Stop/slow
        self.declare_parameter('speed_limit_topic', 'speed_limit')
        self.declare_parameter('enable_slowdown', True)
        self.declare_parameter('slowdown_radius', 2.0)
        self.declare_parameter('slow_speed_limit', 0.10)

        self.declare_parameter('stop_radius', 1.0)
        self.declare_parameter('stop_speed_limit', 0.001)
        self.declare_parameter('stop_hysteresis', 0.15)

        # Debug
        self.declare_parameter('goal_debug_topic', '/approach/goal_pose')

        # ========== Read params ==========
        self.human_topic = str(self.get_parameter('human_topic').value)
        self.global_frame = str(self.get_parameter('global_frame').value)
        self.robot_frame = str(self.get_parameter('robot_frame').value)

        self.nav_action_name = str(self.get_parameter('nav_action_name').value)
        self.path_action_name = str(self.get_parameter('path_action_name').value)

        self.final_distance = float(self.get_parameter('final_distance').value)
        self.candidate_angles_deg = list(self.get_parameter('candidate_angles_deg').value)

        self.allowed_min = math.radians(float(self.get_parameter('allowed_min_deg').value))
        self.allowed_max = math.radians(float(self.get_parameter('allowed_max_deg').value))

        self.behind_guard_radius = float(self.get_parameter('behind_guard_radius').value)
        self.staging_distance = float(self.get_parameter('staging_distance').value)
        self.staging_angle = math.radians(float(self.get_parameter('staging_angle_deg').value))

        self.path_gate_step = float(self.get_parameter('path_gate_step').value)
        self.personal_space_radius = float(self.get_parameter('personal_space_radius').value)

        self.min_goal_period = float(self.get_parameter('min_goal_period').value)
        self.goal_update_dist_thresh = float(self.get_parameter('goal_update_dist_thresh').value)
        self.goal_update_yaw_thresh = math.radians(float(self.get_parameter('goal_update_yaw_thresh_deg').value))

        self.human_update_dist_thresh = float(self.get_parameter('human_update_dist_thresh').value)
        self.human_update_yaw_thresh = math.radians(float(self.get_parameter('human_update_yaw_thresh_deg').value))

        self.rate_hz = float(self.get_parameter('control_rate_hz').value)

        self.speed_limit_topic = str(self.get_parameter('speed_limit_topic').value)
        self.enable_slowdown = bool(self.get_parameter('enable_slowdown').value)
        self.slowdown_radius = float(self.get_parameter('slowdown_radius').value)
        self.slow_speed_limit = float(self.get_parameter('slow_speed_limit').value)

        self.stop_radius = float(self.get_parameter('stop_radius').value)
        self.stop_speed_limit = float(self.get_parameter('stop_speed_limit').value)
        self.stop_hysteresis = float(self.get_parameter('stop_hysteresis').value)

        self.goal_debug_topic = str(self.get_parameter('goal_debug_topic').value)

        # ========== ROS interfaces ==========
        self.human_sub = self.create_subscription(PoseStamped, self.human_topic, self._on_human_pose, 10)
        self.speed_pub = self.create_publisher(SpeedLimit, self.speed_limit_topic, 10)
        self.goal_debug_pub = self.create_publisher(PoseStamped, self.goal_debug_topic, 10)

        self.nav_client = ActionClient(self, NavigateToPose, self.nav_action_name)
        self.path_client = ActionClient(self, ComputePathToPose, self.path_action_name)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ========== State ==========
        self.human: Optional[HumanState] = None

        self.last_sent_goal: Optional[Tuple[float, float, float]] = None
        self.last_goal_time = self.get_clock().now()

        self.last_human_for_goal: Optional[Tuple[float, float, float]] = None

        self.goal_handle = None
        self.active_goal = False
        self.last_nav_status: Optional[int] = None

        self.in_stop_hold = False

        # candidate evaluation pipeline
        self.eval_in_progress = False
        self.eval_candidates: List[Tuple[float, float, float, str]] = []
        self.eval_index = 0
        self.eval_human_snapshot: Optional[HumanState] = None

        self.timer = self.create_timer(1.0 / max(1.0, self.rate_hz), self._tick)

        self.get_logger().info(
            f"ApproachGoalSender ready\n"
            f"  stop_radius={self.stop_radius:.2f}, behind_guard_radius={self.behind_guard_radius:.2f}\n"
            f"  personal_space_radius={self.personal_space_radius:.2f}\n"
            f"  candidates={self.candidate_angles_deg}"
        )

    # ---------- Callbacks ----------
    def _on_human_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        self.human = HumanState(
            x=float(msg.pose.position.x),
            y=float(msg.pose.position.y),
            yaw=float(yaw_from_quat(q.x, q.y, q.z, q.w)),
            stamp_sec=int(msg.header.stamp.sec),
            stamp_nanosec=int(msg.header.stamp.nanosec),
        )

    # ---------- TF ----------
    def _get_robot_pose_map(self) -> Optional[Tuple[float, float, float]]:
        try:
            tfm = self.tf_buffer.lookup_transform(self.global_frame, self.robot_frame, rclpy.time.Time())
            t = tfm.transform.translation
            q = tfm.transform.rotation
            return (float(t.x), float(t.y), float(yaw_from_quat(q.x, q.y, q.z, q.w)))
        except TransformException as ex:
            self.get_logger().warn(f"TF not ready ({self.global_frame}->{self.robot_frame}): {ex}")
            return None

    # ---------- Speed limit ----------
    def _publish_speed_limit(self, mode: str):
        msg = SpeedLimit()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.global_frame
        msg.percentage = False

        if mode == "STOP":
            msg.speed_limit = float(self.stop_speed_limit)   # near-zero
        elif mode == "SLOW":
            msg.speed_limit = float(self.slow_speed_limit)
        else:
            msg.speed_limit = 0.0  # NO LIMIT

        self.speed_pub.publish(msg)

    # ---------- Geometry ----------
    def _clamp_sector(self, angle_rel: float) -> float:
        return max(self.allowed_min, min(self.allowed_max, angle_rel))

    def _compute_goal(self, human: HumanState, dist: float, angle_rel: float) -> Tuple[float, float, float]:
        angle_rel = self._clamp_sector(angle_rel)
        direction = human.yaw + angle_rel
        gx = human.x + dist * math.cos(direction)
        gy = human.y + dist * math.sin(direction)
        gyaw = math.atan2(human.y - gy, human.x - gx)  # face human
        return (gx, gy, gyaw)

    def _is_robot_behind_human(self, rx: float, ry: float, human: HumanState) -> bool:
        ang_h_to_r = math.atan2(ry - human.y, rx - human.x)
        rel = wrap_to_pi(ang_h_to_r - human.yaw)
        return abs(rel) > (math.pi / 2.0)

    # ---------- Replan triggers ----------
    def _human_changed_enough(self, human: HumanState) -> bool:
        if self.last_human_for_goal is None:
            return True
        hx0, hy0, hyaw0 = self.last_human_for_goal
        d = dist2d(human.x, human.y, hx0, hy0)
        dyaw = abs(wrap_to_pi(human.yaw - hyaw0))
        return (d > self.human_update_dist_thresh) or (dyaw > self.human_update_yaw_thresh)

    def _goal_changed_enough(self, new_goal: Tuple[float, float, float]) -> bool:
        if self.last_sent_goal is None:
            return True
        lx, ly, lyaw = self.last_sent_goal
        nx, ny, nyaw = new_goal
        d = dist2d(lx, ly, nx, ny)
        dyaw = abs(wrap_to_pi(nyaw - lyaw))
        return (d > self.goal_update_dist_thresh) or (dyaw > self.goal_update_yaw_thresh)

    def _can_replan_now(self) -> bool:
        now = self.get_clock().now()
        dt = (now - self.last_goal_time).nanoseconds / 1e9
        return dt >= self.min_goal_period

    # ---------- Nav actions ----------
    def _cancel_nav_goal(self):
        if self.goal_handle is None:
            return
        try:
            self.goal_handle.cancel_goal_async()
        except Exception as e:
            self.get_logger().warn(f"Cancel failed: {e}")
        self.goal_handle = None
        self.active_goal = False

    def _send_nav_goal(self, goal_xyyaw: Tuple[float, float, float], label: str):
        if not self.nav_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warn("NavigateToPose server not ready")
            return

        gx, gy, gyaw = goal_xyyaw
        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(gx)
        pose.pose.position.y = float(gy)
        qx, qy, qz, qw = quat_from_yaw(gyaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.get_logger().info(f"[{label}] NavigateToPose -> ({gx:.2f},{gy:.2f}) yaw={math.degrees(gyaw):.1f}")
        fut = self.nav_client.send_goal_async(goal_msg)
        fut.add_done_callback(self._on_nav_goal_response)

        self.last_sent_goal = goal_xyyaw
        self.last_goal_time = self.get_clock().now()
        if self.human is not None:
            self.last_human_for_goal = (self.human.x, self.human.y, self.human.yaw)

        self.active_goal = True
        self.goal_debug_pub.publish(pose)

    def _on_nav_goal_response(self, future):
        gh = future.result()
        if not gh or not gh.accepted:
            self.get_logger().warn("Nav goal rejected")
            self.active_goal = False
            self.goal_handle = None
            self.last_nav_status = None
            return
        self.goal_handle = gh
        res_fut = gh.get_result_async()
        res_fut.add_done_callback(self._on_nav_result)

    def _on_nav_result(self, future):
        status = future.result().status
        self.last_nav_status = status
        # 4=SUCCEEDED, 5=CANCELED, 6=ABORTED
        if status == 4:
            self.get_logger().info("Nav result: SUCCEEDED (4)")
        elif status == 5:
            self.get_logger().warn("Nav result: CANCELED (5)")
        elif status == 6:
            self.get_logger().warn("Nav result: ABORTED (6)")
        else:
            self.get_logger().warn(f"Nav result: status={status}")

        self.active_goal = False
        self.goal_handle = None

    # ---------- Path gate actions ----------
    def _start_eval(self, robot_pose: Tuple[float, float, float], human: HumanState):
        rx, ry, _ = robot_pose
        dist_rh = dist2d(rx, ry, human.x, human.y)
        behind = self._is_robot_behind_human(rx, ry, human)

        cands: List[Tuple[float, float, float, str]] = []
        if behind and dist_rh < self.behind_guard_radius:
            g = self._compute_goal(human, self.staging_distance, self.staging_angle)
            cands.append((g[0], g[1], g[2], "STAGING"))

        for ang_deg in self.candidate_angles_deg:
            g = self._compute_goal(human, self.final_distance, math.radians(float(ang_deg)))
            cands.append((g[0], g[1], g[2], "FINAL"))

        self.eval_candidates = cands
        self.eval_index = 0
        self.eval_in_progress = True
        self.eval_human_snapshot = HumanState(human.x, human.y, human.yaw, human.stamp_sec, human.stamp_nanosec)
        self._eval_next()

    def _eval_next(self):
        if not self.eval_in_progress or self.eval_human_snapshot is None:
            return
        if self.eval_index >= len(self.eval_candidates):
            self.get_logger().warn("No valid candidate (planner failed OR behind/too-close gates rejected all).")
            self.eval_in_progress = False
            return

        if not self.path_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warn("ComputePathToPose server not ready")
            self.eval_in_progress = False
            return

        gx, gy, gyaw, label = self.eval_candidates[self.eval_index]

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.global_frame
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = float(gx)
        goal_pose.pose.position.y = float(gy)
        qx, qy, qz, qw = quat_from_yaw(gyaw)
        goal_pose.pose.orientation.z = qz
        goal_pose.pose.orientation.w = qw

        req = ComputePathToPose.Goal()
        req.goal = goal_pose
        req.planner_id = ""

        fut = self.path_client.send_goal_async(req)
        fut.add_done_callback(self._on_path_goal_response)

    def _on_path_goal_response(self, future):
        gh = future.result()
        if not gh or not gh.accepted:
            self.get_logger().warn(f"Candidate[{self.eval_index}] planner REJECTED goal")
            self.eval_index += 1
            self._eval_next()
            return
        res_fut = gh.get_result_async()
        res_fut.add_done_callback(self._on_path_result)

    def _on_path_result(self, future):
        if not self.eval_in_progress or self.eval_human_snapshot is None:
            return

        human = self.eval_human_snapshot
        result = future.result().result
        path = result.path

        if path is None or len(path.poses) == 0:
            self.get_logger().warn(f"Candidate[{self.eval_index}] planner returned EMPTY path")
            self.eval_index += 1
            self._eval_next()
            return

        # Gate 1: behind near human
        if self._path_goes_behind(path.poses, human, self.behind_guard_radius, self.path_gate_step):
            self.get_logger().warn(f"Candidate[{self.eval_index}] rejected: PATH goes BEHIND human")
            self.eval_index += 1
            self._eval_next()
            return

        # Gate 2: too close near human
        if self._path_too_close(path.poses, human, self.personal_space_radius, self.path_gate_step):
            self.get_logger().warn(f"Candidate[{self.eval_index}] rejected: PATH enters personal space (<{self.personal_space_radius:.2f}m)")
            self.eval_index += 1
            self._eval_next()
            return

        # ACCEPT
        gx, gy, gyaw, label = self.eval_candidates[self.eval_index]
        new_goal = (gx, gy, gyaw)
        self.eval_in_progress = False

        if (not self._can_replan_now()) or (not self._goal_changed_enough(new_goal) and not self._human_changed_enough(human)):
            return

        if self.active_goal and self.goal_handle is not None:
            self._cancel_nav_goal()

        self._send_nav_goal(new_goal, label)

    def _path_goes_behind(self, poses, human: HumanState, near_radius: float, step_m: float) -> bool:
        last_x = poses[0].pose.position.x
        last_y = poses[0].pose.position.y
        for p in poses:
            x = p.pose.position.x
            y = p.pose.position.y
            if dist2d(x, y, last_x, last_y) < step_m:
                continue
            last_x, last_y = x, y

            if dist2d(x, y, human.x, human.y) > near_radius:
                continue

            ang = math.atan2(y - human.y, x - human.x)
            rel = wrap_to_pi(ang - human.yaw)
            if abs(rel) > (math.pi / 2.0):
                return True
        return False

    def _path_too_close(self, poses, human: HumanState, radius: float, step_m: float) -> bool:
        last_x = poses[0].pose.position.x
        last_y = poses[0].pose.position.y
        for p in poses:
            x = p.pose.position.x
            y = p.pose.position.y
            if dist2d(x, y, last_x, last_y) < step_m:
                continue
            last_x, last_y = x, y

            if dist2d(x, y, human.x, human.y) < radius:
                return True
        return False

    # ---------- Main loop ----------
    def _tick(self):
        if self.human is None:
            return

        robot_pose = self._get_robot_pose_map()
        if robot_pose is None:
            return

        rx, ry, _ = robot_pose
        human = self.human
        dist_rh = dist2d(rx, ry, human.x, human.y)

        # STOP latch
        if self.in_stop_hold:
            if dist_rh > (self.stop_radius + self.stop_hysteresis):
                self.in_stop_hold = False
            else:
                self._publish_speed_limit("STOP")
                self._cancel_nav_goal()
                return
        else:
            if dist_rh < self.stop_radius:
                self.in_stop_hold = True
                self._publish_speed_limit("STOP")
                self._cancel_nav_goal()
                return

        # Slow / free
        if self.enable_slowdown and dist_rh < self.slowdown_radius:
            self._publish_speed_limit("SLOW")
        else:
            self._publish_speed_limit("FREE")

        # If evaluating, wait
        if self.eval_in_progress:
            # if human moved a lot while we evaluate, abort eval and restart next tick
            if self._human_changed_enough(human) and self._can_replan_now():
                self.eval_in_progress = False
            else:
                return

        # Replan decision:
        # - human changed enough OR
        # - no active goal OR
        # - last nav aborted OR
        # - time allows and goal likely changes
        need = False
        if self._human_changed_enough(human):
            need = True
        if not self.active_goal:
            need = True
        if self.last_nav_status == 6:  # ABORTED
            need = True

        if not need:
            return
        if not self._can_replan_now():
            return

        self._start_eval(robot_pose, human)


def main():
    rclpy.init()
    node = ApproachGoalSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
