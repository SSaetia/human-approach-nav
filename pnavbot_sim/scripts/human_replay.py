#!/usr/bin/env python3
import math
import csv
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List

import yaml
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile

from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Sample:
    t: float     # seconds from start
    x0: float    # meters (scaled only, NO offset)
    y0: float    # meters (scaled only, NO offset)
    z0: float    # meters (scaled only, NO offset)


# -----------------------------
# Utils
# -----------------------------
def parse_timestamp(ts: str) -> datetime:
    date_part, time_part = ts.split('-')
    hh, mm, ss, ms = time_part.split(':')
    ss = ss.zfill(2)
    ms = ms.zfill(3)
    return datetime.strptime(f"{date_part} {hh}:{mm}:{ss}.{ms}", "%Y/%m/%d %H:%M:%S.%f")


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def yaw_from_velocity(dx: float, dy: float, last_yaw: float, min_speed: float) -> float:
    speed = math.hypot(dx, dy)
    if speed < min_speed:
        return last_yaw
    return math.atan2(dy, dx)


def wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# -----------------------------
# Map (PGM) loader + free-space snapping
# -----------------------------
class OccupancyGridPGM:
    def __init__(self, map_yaml_path: str):
        m = yaml.safe_load(open(map_yaml_path, "r"))
        self.res = float(m["resolution"])
        self.ox, self.oy, _ = m["origin"]
        self.negate = int(m.get("negate", 0))
        self.occ_thresh = float(m.get("occupied_thresh", 0.65))
        self.free_thresh = float(m.get("free_thresh", 0.25))
        pgm_file = m["image"]
        self.pgm_path = os.path.join(os.path.dirname(map_yaml_path), pgm_file)
        self.w, self.h, self.maxval, self.data = self._load_pgm_p5(self.pgm_path)

    @staticmethod
    def _load_pgm_p5(path: str):
        with open(path, "rb") as f:
            tokens = []
            while len(tokens) < 4:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if (not line) or line.startswith(b"#"):
                    continue
                tokens += line.split()

            magic = tokens[0]
            if magic != b"P5":
                raise RuntimeError(f"Only P5 PGM supported, got {magic} in {path}")

            w = int(tokens[1])
            h = int(tokens[2])
            maxval = int(tokens[3])
            data = f.read(w * h)
            if len(data) != w * h:
                raise RuntimeError(f"PGM data size mismatch: expected {w*h}, got {len(data)}")
        return w, h, maxval, data

    def _occ_prob_from_pixel(self, pix: int) -> float:
        if self.negate == 0:
            return (255 - pix) / 255.0
        else:
            return pix / 255.0

    def is_free_pixel(self, pix: int) -> bool:
        p = self._occ_prob_from_pixel(pix)
        if p > self.occ_thresh:
            return False
        if p < self.free_thresh:
            return True
        return False  # unknown treated as not-free

    def world_to_map(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        mx = int((x - self.ox) / self.res)
        my = int((y - self.oy) / self.res)
        px = mx
        py = (self.h - 1) - my
        if 0 <= px < self.w and 0 <= py < self.h:
            return px, py
        return None

    def map_to_world(self, px: int, py: int) -> Tuple[float, float]:
        my = (self.h - 1) - py
        mx = px
        x = self.ox + (mx + 0.5) * self.res
        y = self.oy + (my + 0.5) * self.res
        return x, y

    def snap_to_nearest_free(self, x: float, y: float, radius_m: float) -> Tuple[float, float, bool]:
        ij = self.world_to_map(x, y)
        if ij is None:
            return x, y, False
        px, py = ij

        if self.is_free_pixel(self.data[py * self.w + px]):
            return x, y, False

        r_px = max(1, int(radius_m / self.res))

        for r in range(1, r_px + 1):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    qx, qy = px + dx, py + dy
                    if 0 <= qx < self.w and 0 <= qy < self.h:
                        if self.is_free_pixel(self.data[qy * self.w + qx]):
                            xn, yn = self.map_to_world(qx, qy)
                            return xn, yn, True
            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    qx, qy = px + dx, py + dy
                    if 0 <= qx < self.w and 0 <= qy < self.h:
                        if self.is_free_pixel(self.data[qy * self.w + qx]):
                            xn, yn = self.map_to_world(qx, qy)
                            return xn, yn, True

        return x, y, False


# -----------------------------
# ROS2 Node
# -----------------------------
class HumanReplay(Node):
    def __init__(self):
        super().__init__("human_replay")

        # ---- Parameters (can be changed at runtime)
        self.declare_parameter("csv_path", "/home/hayashi/pat_ws/src/pnavbot/pnavbot_sim/include/footpoints_log.csv")
        self.declare_parameter("person_id", "koko")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("human_frame", "human_koko")

        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("playback_speed", 1.0)
        self.declare_parameter("loop", True)

        # scale is applied to raw CSV; offsets are applied at publish time (dynamic)
        self.declare_parameter("scale", 0.003)
        self.declare_parameter("x_offset", 0.15131719657875653)
        self.declare_parameter("y_offset", 0.26813119207875835)

        self.declare_parameter("max_step", 0.03)
        self.declare_parameter("min_speed_for_yaw", 0.02)
        self.declare_parameter("yaw_noise_std", 0.02)
        self.declare_parameter("yaw_smoothing", 0.2)

        self.declare_parameter("map_yaml", "/home/hayashi/pat_ws/src/pnavbot/pnavbot_sim/maps/anna_map.yaml")
        self.declare_parameter("snap_to_free", True)
        self.declare_parameter("snap_radius_m", 0.8)

        self.declare_parameter("publish_tf", True)
        self.declare_parameter("publish_marker", True)

        # ---- Load fixed params / initial state
        self.csv_path = str(self.get_parameter("csv_path").value)
        self.person_id = str(self.get_parameter("person_id").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.human_frame = str(self.get_parameter("human_frame").value)

        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.publish_marker = bool(self.get_parameter("publish_marker").value)

        # Dynamic params (will be updated by param callback)
        self._sync_dynamic_params_from_server()

        # ---- Load map for snapping
        self.map_grid = None
        if self.snap_to_free:
            self.map_grid = OccupancyGridPGM(self.map_yaml)
            self.get_logger().info(
                f"Loaded map: {self.map_grid.w}x{self.map_grid.h}, res={self.map_grid.res:.6f}, "
                f"origin=({self.map_grid.ox:.3f},{self.map_grid.oy:.3f})"
            )

        # ---- Publishers
        self.pose_pub = self.create_publisher(PoseStamped, "/human/pose", 10)
        self.marker_pub = self.create_publisher(Marker, "/human/marker", 10) if self.publish_marker else None
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        # ---- Service: reset playback (respawn effect)
        self.reset_srv = self.create_service(Trigger, "reset", self._on_reset)

        # ---- Parameter callback: apply changes live
        self.add_on_set_parameters_callback(self._on_param_update)

        # ---- Load samples (scaled only; offsets applied later)
        self.samples = self._load_samples_scaled_only(self.csv_path, self.person_id, self.scale)
        if len(self.samples) < 2:
            raise RuntimeError(f"Not enough samples for person_id='{self.person_id}' in {self.csv_path}")
        self.get_logger().info(f"Loaded {len(self.samples)} samples for person_id='{self.person_id}'")

        # RNG
        self.rng = random.Random(7)

        # Init state
        self._reset_playback_state(hard=True)

        # Timer
        dt = 1.0 / max(1e-6, self.rate_hz)
        self.timer = self.create_timer(dt, self.on_timer)

        self.get_logger().info("Ready. You can change params live + call /human_replay/reset.")

    # -------- param handling --------
    def _sync_dynamic_params_from_server(self):
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.playback_speed = float(self.get_parameter("playback_speed").value)
        self.loop = bool(self.get_parameter("loop").value)

        self.scale = float(self.get_parameter("scale").value)
        self.x_offset = float(self.get_parameter("x_offset").value)
        self.y_offset = float(self.get_parameter("y_offset").value)

        self.max_step = float(self.get_parameter("max_step").value)
        self.min_speed_for_yaw = float(self.get_parameter("min_speed_for_yaw").value)
        self.yaw_noise_std = float(self.get_parameter("yaw_noise_std").value)
        self.yaw_smoothing = float(self.get_parameter("yaw_smoothing").value)

        self.map_yaml = str(self.get_parameter("map_yaml").value)
        self.snap_to_free = bool(self.get_parameter("snap_to_free").value)
        self.snap_radius_m = float(self.get_parameter("snap_radius_m").value)

    def _on_param_update(self, params: List[Parameter]):
        # Apply updates safely
        changed_scale = False
        for p in params:
            if p.name == "rate_hz":
                self.rate_hz = float(p.value)
            elif p.name == "playback_speed":
                self.playback_speed = float(p.value)
            elif p.name == "loop":
                self.loop = bool(p.value)
            elif p.name == "x_offset":
                self.x_offset = float(p.value)
            elif p.name == "y_offset":
                self.y_offset = float(p.value)
            elif p.name == "yaw_noise_std":
                self.yaw_noise_std = float(p.value)
            elif p.name == "yaw_smoothing":
                self.yaw_smoothing = float(p.value)
            elif p.name == "max_step":
                self.max_step = float(p.value)
            elif p.name == "min_speed_for_yaw":
                self.min_speed_for_yaw = float(p.value)
            elif p.name == "snap_radius_m":
                self.snap_radius_m = float(p.value)
            elif p.name == "snap_to_free":
                self.snap_to_free = bool(p.value)
            elif p.name == "scale":
                self.scale = float(p.value)
                changed_scale = True

        # If scale changed, we must reload scaled samples
        if changed_scale:
            self.samples = self._load_samples_scaled_only(self.csv_path, self.person_id, self.scale)
            self.get_logger().warn("scale changed -> reloaded samples. Call /human_replay/reset to restart cleanly.")

        return rclpy.parameter.SetParametersResult(successful=True)

    # -------- reset service --------
    def _on_reset(self, request, response):
        self._reset_playback_state(hard=True)
        response.success = True
        response.message = "Reset OK"
        return response

    def _reset_playback_state(self, hard: bool = False):
        self.start_time_ros = self.get_clock().now()
        self.i = 0

        # start at first sample + current offsets
        x0 = self.samples[0].x0 + self.x_offset
        y0 = self.samples[0].y0 + self.y_offset

        if self.map_grid and self.snap_to_free:
            xs, ys, changed = self.map_grid.snap_to_nearest_free(x0, y0, self.snap_radius_m)
            if changed:
                self.get_logger().warn(f"Reset: start was not free -> snapped to ({xs:.3f},{ys:.3f})")
                x0, y0 = xs, ys

        self.last_x = x0
        self.last_y = y0
        self.last_yaw = 0.0

    # -------- loading & interpolation --------
    def _load_samples_scaled_only(self, csv_path: str, person_id: str, scale: float) -> List[Sample]:
        rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("person_id", "") != person_id:
                    continue
                ts = parse_timestamp(r["timestamp"])
                rows.append((ts, float(r["x"]), float(r["y"]), float(r["z"])))

        rows.sort(key=lambda x: x[0])
        t0 = rows[0][0]

        samples = []
        for (ts, x_raw, y_raw, z_raw) in rows:
            t = (ts - t0).total_seconds()
            x0 = x_raw * scale
            y0 = y_raw * scale
            z0 = z_raw * scale
            samples.append(Sample(t=t, x0=x0, y0=y0, z0=z0))
        return samples

    def _interpolate_scaled_only(self, t_play: float) -> Tuple[float, float]:
        while (self.i + 1) < len(self.samples) and self.samples[self.i + 1].t < t_play:
            self.i += 1
        i2 = min(self.i + 1, len(self.samples) - 1)

        s1 = self.samples[self.i]
        s2 = self.samples[i2]
        if s2.t <= s1.t:
            u = 0.0
        else:
            u = (t_play - s1.t) / (s2.t - s1.t)

        x0 = lerp(s1.x0, s2.x0, u)
        y0 = lerp(s1.y0, s2.y0, u)
        return x0, y0

    # -------- main loop --------
    def on_timer(self):
        now = self.get_clock().now()
        elapsed = (now - self.start_time_ros).nanoseconds * 1e-9
        t_play = elapsed * self.playback_speed

        t_end = self.samples[-1].t
        if t_play > t_end:
            if not self.loop:
                return
            self.start_time_ros = now
            self.i = 0
            t_play = 0.0

        # interpolate (scaled only) + apply current offsets LIVE
        x0, y0 = self._interpolate_scaled_only(t_play)
        x_tgt = x0 + self.x_offset
        y_tgt = y0 + self.y_offset

        # speed cap
        dx = x_tgt - self.last_x
        dy = y_tgt - self.last_y
        step = math.hypot(dx, dy)
        if step > self.max_step and step > 1e-9:
            ratio = self.max_step / step
            x_tgt = self.last_x + dx * ratio
            y_tgt = self.last_y + dy * ratio

        # snap to free
        if self.map_grid and self.snap_to_free:
            xs, ys, changed = self.map_grid.snap_to_nearest_free(x_tgt, y_tgt, self.snap_radius_m)
            if changed:
                x_tgt, y_tgt = xs, ys

        dx = x_tgt - self.last_x
        dy = y_tgt - self.last_y

        yaw_meas = yaw_from_velocity(dx, dy, self.last_yaw, self.min_speed_for_yaw)
        yaw_meas += self.rng.gauss(0.0, self.yaw_noise_std)

        a = max(0.0, min(1.0, self.yaw_smoothing))
        yaw = wrap_pi((1.0 - a) * self.last_yaw + a * yaw_meas)

        self.last_x, self.last_y, self.last_yaw = x_tgt, y_tgt, yaw

        # publish pose
        msg = PoseStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = x_tgt
        msg.pose.position.y = y_tgt
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = math.sin(yaw * 0.5)
        msg.pose.orientation.w = math.cos(yaw * 0.5)
        self.pose_pub.publish(msg)

        # TF
        if self.tf_broadcaster:
            tfm = TransformStamped()
            tfm.header = msg.header
            tfm.child_frame_id = self.human_frame
            tfm.transform.translation.x = x_tgt
            tfm.transform.translation.y = y_tgt
            tfm.transform.translation.z = 0.0
            tfm.transform.rotation = msg.pose.orientation
            self.tf_broadcaster.sendTransform(tfm)

        # marker
        if self.marker_pub:
            mk = Marker()
            mk.header = msg.header
            mk.ns = "human"
            mk.id = 0
            mk.type = Marker.ARROW
            mk.action = Marker.ADD
            mk.pose = msg.pose
            mk.scale.x = 0.6
            mk.scale.y = 0.10
            mk.scale.z = 0.10
            mk.color.a = 1.0
            mk.color.r = 1.0
            mk.color.g = 0.5
            mk.color.b = 0.2
            self.marker_pub.publish(mk)


def main():
    rclpy.init()
    node = HumanReplay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
