#############################################################################################
##  mapper.py
##  For Viz of footpoints
##  Contains portable dataclass Footpoints that is standard for processing of footpoint data
##  Cleaning of bottom/top out of bounds area, visualisation fcts
#############################################################################################

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2
from PIL import Image
import io
import open3d as o3d
import math
import yaml
from collections import deque, defaultdict
from typing import Dict, List, Optional

from dataclasses import dataclass

import yaml #this for nav2

@dataclass
class Footpoint:
    person_id: int
    x: int
    y: int
    z: int
    framestamp: int

class mapVisualizer:
    def __init__(self, 
                 pcd_path, 
                 num_people,
                 img_size=(800, 800),
                 z_min=25, z_max=300,
                 stretch_xy=3.0):
        pcd_read = o3d.io.read_point_cloud(pcd_path)
        pcd = np.array(pcd_read.points)

        # 1) Stretch XY (unitless -> you decide scale)
        pcd = self.scale_points_xy(pcd, stretch_xy)

        # 2) Filter Z only (Anna's intention)
        self.map_points = self.clean_pcd_z_only(pcd, z_min, z_max)

        # Setup
        self.dynamic_objects = {}
        self.img_size = img_size
        self.num_people = num_people

        self.colors = [
            (255, 0, 0),(0, 255, 0),(0, 0, 255),(255, 255, 0),
            (255, 0, 255),(0, 255, 255),(128, 0, 128),(255, 165, 0),
            (0, 128, 128),(128, 128, 0),(255, 100, 100),(100, 255, 244),
            (100, 64, 100),(50, 200, 100),
        ]
        self.enough_colors_for_people_check()

        # Map bounds in "scaled units"
        self.x_min, self.x_max = self.map_points[:, 0].min(), self.map_points[:, 0].max()
        self.y_min, self.y_max = self.map_points[:, 1].min(), self.map_points[:, 1].max()

        # pixels per unit
        self.x_scale = (img_size[0] - 20) / (self.x_max - self.x_min)
        self.y_scale = (img_size[1] - 20) / (self.y_max - self.y_min)

        self._create_base_image()

    
    def enough_colors_for_people_check(self):
        if len(self.colors) < self.num_people:
            raise Exception (f"mapper.mapViz: Not enough colours ({len(self.colors)}) for {self.num_people} people.\nResolve this issue by adding more colours in the mapViz constructor.")
 
    def scale_points_xy(self, points: np.ndarray, scale: float) -> np.ndarray:
        out = points.copy()
        out[:, 0] *= float(scale)
        out[:, 1] *= float(scale)
        return out

    def clean_pcd_z_only(self, pcd: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
        """Anna's intention: remove floor/ceiling noise by z filtering only."""
        mask = (pcd[:, 2] >= z_min) & (pcd[:, 2] <= z_max)
        pts = pcd[mask]
        print(f"Z filter: {z_min} <= z <= {z_max} | kept {len(pts)}/{len(pcd)}")
        return pts

    def compute_resolution(self, meters_per_unit: float) -> float:
        """
        If your scaled points are still in 'units', you must define meters_per_unit.
        resolution (m/pixel) = (m/unit) / (px/unit)
        """
        px_per_unit = 0.5 * (self.x_scale + self.y_scale)
        return float(meters_per_unit / px_per_unit)


    def export_nav2_map(self, out_prefix: str, resolution_m_per_px: float, meters_per_unit: float):
        gray = cv2.cvtColor(self.base_img, cv2.COLOR_BGR2GRAY)
        img = np.flipud(gray)

        pgm_path = out_prefix + ".pgm"
        yaml_path = out_prefix + ".yaml"

        Image.fromarray(img).save(pgm_path)

        # Convert origin from "units" to meters
        origin = [float(self.x_min * meters_per_unit), float(self.y_min * meters_per_unit), 0.0]

        data = {
            "image": pgm_path.split("/")[-1],
            "resolution": float(resolution_m_per_px),
            "origin": origin,
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.25,
        }
        with open(yaml_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        print("Saved:", pgm_path)
        print("Saved:", yaml_path)
        print("origin(m):", origin, "resolution(m/px):", resolution_m_per_px)




    # def clean_pcd(self, pcd, l_x=-3000, u_x=1000, l_y=-3000, u_y=3500, l_z=25, u_z=300):
    #     try:
    #         mask = (
    #             (pcd[:, 0] >= l_x) & (pcd[:, 0] <= u_x) &
    #             (pcd[:, 1] >= l_y) & (pcd[:, 1] <= u_y) &
    #             (pcd[:, 2] >= l_z) & (pcd[:, 2] <= u_z)
    #         )
    #         points = pcd[mask]
    #         print(f"filtered map for bounds: \n {l_x} < x < {u_x} \n {l_y} < y < {u_y} \n {l_z} < z < {u_z}")
    #         print("kept points:", len(points), "/", len(pcd))
    #         return points
    #     except Exception as e:
    #         print(f"mapper: error limiting points in pcd: {e}")
    #         return pcd


    def _create_base_image(self):
        """Create base image with static map points"""
        self.base_img = np.ones((self.img_size[1], self.img_size[0], 3), dtype=np.uint8) * 255
        
        for point in self.map_points[::10]:  # Subsample for performance
            x_pixel = int((point[0] - self.x_min) * self.x_scale) + 10
            y_pixel = int((point[1] - self.y_min) * self.y_scale) + 10
            
            if 0 <= x_pixel < self.img_size[0] and 0 <= y_pixel < self.img_size[1]:
                cv2.circle(self.base_img, (x_pixel, y_pixel), 1, (0, 0, 0), -1)
    
    def world_to_pixel(self, x, y):
        """Convert world coordinates to pixel coordinates"""
        x_pixel = int((x - self.x_min) * self.x_scale) + 10
        y_pixel = int((y - self.y_min) * self.y_scale) + 10
        return x_pixel, y_pixel
    

    def add_dynamic_points(self, footpoints, max_trail_length=50):
        img = self.base_img.copy()
        person_ids = list(set(fp.person_id for fp in footpoints))
        
        for footpoint in footpoints:
            person_id = footpoint.person_id
            x = footpoint.x
            y = footpoint.y
            z = footpoint.z
            
            # Get color index based on person_id
            if person_id not in self.dynamic_objects:
                # Assign color based on person_id index or hash
                try:
                    color_index = person_ids.index(person_id) % len(self.colors)
                except ValueError:
                    # Fallback: use hash of person_id if not in current list
                    color_index = hash(person_id) % len(self.colors)
                
                self.dynamic_objects[person_id] = {
                    'history': deque(maxlen=max_trail_length),
                    'color': self.colors[color_index],
                    'person_id': person_id
                }
            
            obj = self.dynamic_objects[person_id]
            obj['history'].append((x, y))
        
        # Draw all dynamic objects
        for person_id, obj in self.dynamic_objects.items():
            if len(obj['history']) == 0:
                continue
            
            # Draw trail
            if len(obj['history']) > 1:
                points = []
                for x, y in obj['history']:
                    px, py = self.world_to_pixel(x, y)
                    if 0 <= px < self.img_size[0] and 0 <= py < self.img_size[1]:
                        points.append((px, py))
                
                if len(points) > 1:
                    points_array = np.array(points, np.int32)
                    cv2.polylines(img, [points_array], False, obj['color'], 2)
            
            # Draw current position
            if obj['history']:
                x, y = obj['history'][-1]
                px, py = self.world_to_pixel(x, y)
                if 0 <= px < self.img_size[0] and 0 <= py < self.img_size[1]:
                    cv2.circle(img, (px, py), 5, obj['color'], -1)
                    
                    # Show person ID
                    label = f"{obj['person_id']}"
                    cv2.putText(img, label, (px+10, py-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, obj['color'], 1)
        
        return img
        
    def add_static_trajectories(self, footpoints):
        img = self.base_img.copy()
        
        # Group by person_id
        trajectories = {}
        for fp in footpoints:
            if fp.person_id not in trajectories:
                trajectories[fp.person_id] = []
            trajectories[fp.person_id].append((fp.x, fp.y, fp.framestamp))
        
        # Sort each trajectory by framestamp
        for person_id in trajectories:
            trajectories[person_id].sort(key=lambda x: x[2])
        
        # Assign colors
        person_ids = list(trajectories.keys())
        
        # Draw each trajectory
        for idx, (person_id, points) in enumerate(trajectories.items()):
            color = self.colors[idx % len(self.colors)]
            
            # Convert to pixel coordinates
            pixel_points = []
            for x, y, _ in points:
                px, py = self.world_to_pixel(x, y)
                if 0 <= px < self.img_size[0] and 0 <= py < self.img_size[1]:
                    pixel_points.append((px, py))
            
            # Draw trajectory line
            if len(pixel_points) > 1:
                points_array = np.array(pixel_points, np.int32)
                cv2.polylines(img, [points_array], False, color, 2)
            
            # Draw start and end points
            if pixel_points:
                # Start point (green)
                cv2.circle(img, pixel_points[0], 7, (0, 255, 0), -1)
                # End point (red)
                cv2.circle(img, pixel_points[-1], 7, (0, 0, 255), -1)
                
                # Label
                cv2.putText(img, f"{person_id}", 
                        (pixel_points[-1][0]+10, pixel_points[-1][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img

    def visualise_pcd_2d(self):
        xy_points = self.map_points[:, :2]
        plt.figure(figsize=(10, 10))
        plt.scatter(xy_points[:, 0], xy_points[:, 1], s=0.5, c='black')
        plt.title("2D Projection of Point Cloud")
        plt.axis("equal")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()




if __name__ == "__main__":
    pcd_path = "/home/hayashi/pat_ws/src/pnavbot/pnavbot_perception/include/cleanedCloud.pcd"

    # Decide scaling meaning:
    # Example: 1000 units should be 3 meters => meters_per_unit = 0.003
    meters_per_unit = 0.003

    viz = mapVisualizer(
        pcd_path=pcd_path,
        num_people=5,
        img_size=(2000, 2000),
        z_min=25, z_max=300,     # keep same as Anna for now
        stretch_xy=3.0           # your "make room 3x bigger"
    )

    # Compute nav2 resolution properly
    resolution = viz.compute_resolution(meters_per_unit)

    # Export files
    viz.export_nav2_map("/home/hayashi/maps/anna_map", resolution, meters_per_unit)

    # Optional visualize
    viz.visualise_pcd_2d()
#############################################################################################