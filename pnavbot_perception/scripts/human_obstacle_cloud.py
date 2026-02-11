#!/usr/bin/env python3
import math
import struct
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    # planar yaw from quaternion
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def rotate2d(x: float, y: float, yaw: float) -> Tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (c * x - s * y, s * x + c * y)


class HumanObstacleCloud(Node):
    """
    Subscribe human pose (usually in map frame) and publish a ring PointCloud2
    in odom frame so local_costmap (global_frame=odom) can consume it reliably.

    - Input:  /human/pose (PoseStamped)
    - Output: /human/obstacles (PointCloud2) in odom frame, BEST_EFFORT QoS
    """

    def __init__(self):
        super().__init__("human_obstacle_cloud")

        # ---- Parameters
        self.declare_parameter("input_pose_topic", "/human/pose")
        self.declare_parameter("output_cloud_topic", "/human/obstacles")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")

        # Output shape (ring around human)
        self.declare_parameter("ring_radius_m", 1.0)
        self.declare_parameter("ring_points", 36)
        self.declare_parameter("ring_z", 0.0)

        # Optional "behind" extension (simple: add a second ring behind the human yaw)
        # If you don't want it, set enable_behind := false
        self.declare_parameter("enable_behind", False)
        self.declare_parameter("behind_dist_m", 3.0)
        self.declare_parameter("behind_radius_m", 0.6)
        self.declare_parameter("behind_points", 18)

        # TF lookup timeout
        self.declare_parameter("tf_timeout_s", 0.1)

        # ---- Read params
        self.input_pose_topic = str(self.get_parameter("input_pose_topic").value)
        self.output_cloud_topic = str(self.get_parameter("output_cloud_topic").value)

        self.map_frame = str(self.get_parameter("map_frame").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)

        self.ring_radius_m = float(self.get_parameter("ring_radius_m").value)
        self.ring_points = int(self.get_parameter("ring_points").value)
        self.ring_z = float(self.get_parameter("ring_z").value)

        self.enable_behind = bool(self.get_parameter("enable_behind").value)
        self.behind_dist_m = float(self.get_parameter("behind_dist_m").value)
        self.behind_radius_m = float(self.get_parameter("behind_radius_m").value)
        self.behind_points = int(self.get_parameter("behind_points").value)

        self.tf_timeout_s = float(self.get_parameter("tf_timeout_s").value)

        # ---- TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- QoS (match Nav2 obstacle_layer subscription: BEST_EFFORT)
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---- Pub/Sub
        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.input_pose_topic,
            self.on_pose,
            10,  # pose is not "sensor" typically; reliable is OK here
        )
        self.cloud_pub = self.create_publisher(PointCloud2, self.output_cloud_topic, sensor_qos)

        self.get_logger().info(
            f"Listening: {self.input_pose_topic} -> Publishing: {self.output_cloud_topic} (frame={self.odom_frame}, BEST_EFFORT)\n"
            f"Ring: r={self.ring_radius_m:.2f} n={self.ring_points}  Behind: {self.enable_behind}"
        )

    def _lookup_odom_T_map(self, stamp) -> TransformStamped | None:
        try:
            return self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.map_frame,
                stamp,
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout_s),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed ({self.odom_frame}<-{self.map_frame}): {e}")
            return None

    def _transform_point_map_to_odom(self, tfm: TransformStamped, x_m: float, y_m: float, z_m: float) -> Tuple[float, float, float]:
        tx = tfm.transform.translation.x
        ty = tfm.transform.translation.y
        tz = tfm.transform.translation.z

        q = tfm.transform.rotation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        xr, yr = rotate2d(x_m, y_m, yaw)
        return (xr + tx, yr + ty, z_m + tz)

    def _make_ring_points(self, cx: float, cy: float, r: float, n: int, z: float) -> List[Tuple[float, float, float]]:
        pts = []
        n = max(3, int(n))
        for k in range(n):
            ang = 2.0 * math.pi * (k / float(n))
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang), z))
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
        # We assume pose_msg.header.frame_id is map (or compatible)
        stamp = rclpy.time.Time.from_msg(pose_msg.header.stamp)

        tfm = self._lookup_odom_T_map(stamp)
        if tfm is None:
            return  # do not publish wrong data

        # Human center in map
        hx_m = pose_msg.pose.position.x
        hy_m = pose_msg.pose.position.y

        # Main ring in map -> transform to odom
        ring_map = self._make_ring_points(hx_m, hy_m, self.ring_radius_m, self.ring_points, self.ring_z)
        ring_odom = [self._transform_point_map_to_odom(tfm, x, y, z) for (x, y, z) in ring_map]

        all_pts = ring_odom

        # Optional: add a "behind" bubble (placed behind the human orientation)
        if self.enable_behind:
            q = pose_msg.pose.orientation
            human_yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

            # behind center in map frame
            bx = hx_m - self.behind_dist_m * math.cos(human_yaw)
            by = hy_m - self.behind_dist_m * math.sin(human_yaw)

            behind_map = self._make_ring_points(bx, by, self.behind_radius_m, self.behind_points, self.ring_z)
            behind_odom = [self._transform_point_map_to_odom(tfm, x, y, z) for (x, y, z) in behind_map]
            all_pts = all_pts + behind_odom

        cloud = self._build_cloud_msg(pose_msg.header.stamp, self.odom_frame, all_pts)
        self.cloud_pub.publish(cloud)


def main():
    rclpy.init()
    node = HumanObstacleCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
