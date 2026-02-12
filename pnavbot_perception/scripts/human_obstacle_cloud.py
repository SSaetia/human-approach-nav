#!/usr/bin/env python3
import math
import struct
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


def wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quat(yaw: float):
    """Return (x,y,z,w) quaternion for planar yaw."""
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def rotate2d(x: float, y: float, yaw: float) -> Tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (c * x - s * y, s * x + c * y)


class HumanObstacleCloud(Node):
    """
    Subscribe /human/pose (PoseStamped in map), publish /human/obstacles (PointCloud2) in base_link.

    Key idea (fix shifting):
      1) Transform the human pose into base_link using TF (latest).
      2) Generate the disk + behind wedge directly in base_link coordinates.
      3) Publish cloud stamped with now() so TF(time=now) is consistent.

    This makes obstacle_range filtering correct (sensor origin == robot) and removes TF-time mismatch drift.
    """

    def __init__(self):
        super().__init__("human_obstacle_cloud")

        # Topics / frames
        self.declare_parameter("input_pose_topic", "/human/pose")
        self.declare_parameter("output_cloud_topic", "/human/obstacles")
        self.declare_parameter("output_frame", "base_link")   # robot-mounted frame

        # Filled disk around human (stop zone)
        self.declare_parameter("body_radius_m", 0.8)
        self.declare_parameter("body_point_spacing_m", 0.12)

        # Behind wedge region (avoid going behind human)
        self.declare_parameter("enable_behind_zone", True)
        self.declare_parameter("behind_dist_m", 1.5)
        self.declare_parameter("behind_half_angle_deg", 90.0)   # back +/- 90 deg
        self.declare_parameter("behind_radial_step_m", 0.20)
        self.declare_parameter("behind_angular_step_deg", 8.0)
        self.declare_parameter("behind_min_radius_m", 0.6)

        # Z for points
        self.declare_parameter("z", 0.0)

        # TF lookup
        self.declare_parameter("tf_timeout_s", 0.2)

        # Read params
        self.input_pose_topic = str(self.get_parameter("input_pose_topic").value)
        self.output_cloud_topic = str(self.get_parameter("output_cloud_topic").value)
        self.output_frame = str(self.get_parameter("output_frame").value)

        self.body_radius_m = float(self.get_parameter("body_radius_m").value)
        self.body_point_spacing_m = float(self.get_parameter("body_point_spacing_m").value)

        self.enable_behind_zone = bool(self.get_parameter("enable_behind_zone").value)
        self.behind_dist_m = float(self.get_parameter("behind_dist_m").value)
        self.behind_half_angle_deg = float(self.get_parameter("behind_half_angle_deg").value)
        self.behind_radial_step_m = float(self.get_parameter("behind_radial_step_m").value)
        self.behind_angular_step_deg = float(self.get_parameter("behind_angular_step_deg").value)
        self.behind_min_radius_m = float(self.get_parameter("behind_min_radius_m").value)

        self.z = float(self.get_parameter("z").value)
        self.tf_timeout_s = float(self.get_parameter("tf_timeout_s").value)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # QoS: RELIABLE so RViz sees it; Nav2 can still subscribe.
        cloud_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.pose_sub = self.create_subscription(PoseStamped, self.input_pose_topic, self.on_pose, 10)
        self.cloud_pub = self.create_publisher(PointCloud2, self.output_cloud_topic, cloud_qos)

        self.get_logger().info(
            f"Sub: {self.input_pose_topic} -> Pub: {self.output_cloud_topic}\n"
            f"Output frame: {self.output_frame}\n"
            f"Disk r={self.body_radius_m:.2f} spacing={self.body_point_spacing_m:.2f}\n"
            f"Behind zone={self.enable_behind_zone} dist={self.behind_dist_m:.2f} half_angle={self.behind_half_angle_deg:.1f}deg"
        )

    def _lookup(self, target_frame: str, source_frame: str) -> Optional[TransformStamped]:
        """Lookup latest transform target<-source (Time(0)) to keep consistent with cloud stamped now()."""
        timeout = rclpy.duration.Duration(seconds=self.tf_timeout_s)
        try:
            return self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time(), timeout=timeout)
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed ({target_frame}<-{source_frame}): {e}")
            return None

    def _apply_tf_xy_yaw(
        self, tfm: TransformStamped, x_s: float, y_s: float, yaw_s: float
    ) -> Tuple[float, float, float]:
        """
        Apply 2D transform to (x,y,yaw):
          p_t = R * p_s + t
          yaw_t = yaw_tf + yaw_s
        """
        tx = tfm.transform.translation.x
        ty = tfm.transform.translation.y

        q = tfm.transform.rotation
        yaw_tf = quat_to_yaw(q.x, q.y, q.z, q.w)

        xr, yr = rotate2d(x_s, y_s, yaw_tf)
        x_t = xr + tx
        y_t = yr + ty
        yaw_t = wrap_pi(yaw_tf + yaw_s)
        return x_t, y_t, yaw_t

    def _make_filled_disk_points(self, cx: float, cy: float, r: float, spacing: float, z: float) -> List[Tuple[float, float, float]]:
        pts: List[Tuple[float, float, float]] = []
        spacing = max(0.03, float(spacing))
        r = max(0.01, float(r))

        x = -r
        while x <= r + 1e-9:
            y = -r
            while y <= r + 1e-9:
                if (x * x + y * y) <= (r * r):
                    pts.append((cx + x, cy + y, z))
                y += spacing
            x += spacing
        return pts

    def _make_filled_sector_points(
        self,
        cx: float,
        cy: float,
        yaw_center: float,
        half_angle_deg: float,
        r_min: float,
        r_max: float,
        radial_step: float,
        angular_step_deg: float,
        z: float
    ) -> List[Tuple[float, float, float]]:
        pts: List[Tuple[float, float, float]] = []

        half = math.radians(max(0.0, half_angle_deg))
        ang_step = math.radians(max(1.0, angular_step_deg))
        r_min = max(0.0, r_min)
        r_max = max(r_min, r_max)
        radial_step = max(0.05, radial_step)

        a = -half
        while a <= half + 1e-9:
            yaw = wrap_pi(yaw_center + a)
            rr = r_min
            while rr <= r_max + 1e-9:
                pts.append((cx + rr * math.cos(yaw), cy + rr * math.sin(yaw), z))
                rr += radial_step
            a += ang_step
        return pts

    def _build_cloud_msg(self, stamp_msg, frame_id: str, pts: List[Tuple[float, float, float]]) -> PointCloud2:
        msg = PointCloud2()
        msg.header.stamp = stamp_msg
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = len(pts)
        msg.is_bigendian = False
        msg.is_dense = True

        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width

        buf = bytearray()
        for (x, y, z) in pts:
            buf += struct.pack("<fff", float(x), float(y), float(z))
        msg.data = bytes(buf)
        return msg

    def on_pose(self, pose_msg: PoseStamped):
        # Use latest TF and stamp cloud with now() to keep TF-time consistent.
        now = self.get_clock().now()

        source_frame = pose_msg.header.frame_id
        if not source_frame:
            self.get_logger().warn("human pose has empty frame_id, skipping")
            return

        tfm = self._lookup(self.output_frame, source_frame)
        if tfm is None:
            return

        # human in source frame
        hx_s = pose_msg.pose.position.x
        hy_s = pose_msg.pose.position.y
        q = pose_msg.pose.orientation
        face_yaw_s = quat_to_yaw(q.x, q.y, q.z, q.w)

        # transform human (x,y,yaw) into base_link
        hx, hy, face_yaw = self._apply_tf_xy_yaw(tfm, hx_s, hy_s, face_yaw_s)
        back_yaw = wrap_pi(face_yaw + math.pi)

        # generate obstacles directly in base_link coords
        pts = self._make_filled_disk_points(hx, hy, self.body_radius_m, self.body_point_spacing_m, self.z)

        if self.enable_behind_zone:
            pts += self._make_filled_sector_points(
                cx=hx, cy=hy,
                yaw_center=back_yaw,
                half_angle_deg=self.behind_half_angle_deg,
                r_min=self.behind_min_radius_m,
                r_max=self.behind_dist_m,
                radial_step=self.behind_radial_step_m,
                angular_step_deg=self.behind_angular_step_deg,
                z=self.z
            )

        cloud = self._build_cloud_msg(now.to_msg(), self.output_frame, pts)
        self.cloud_pub.publish(cloud)


def main():
    rclpy.init()
    node = HumanObstacleCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
