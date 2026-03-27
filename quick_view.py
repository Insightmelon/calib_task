import os
import re
import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk


def radar_to_lidar(points_R,R_RL,t_RL):
    """Transform radar point cloud (Nx3) into lidar frame."""
    P = np.asarray(points_R, dtype=float).reshape(-1, 3)
    return (R_RL @ P.T).T + t_RL

def load_points_from_csv(path):
    df = pd.read_csv(path)
    points = df[['x', 'y', 'z']].to_numpy()
    # filter within 10m radius
    mask = np.sum(points**2, axis=1) <= 100
    return points[mask]

def get_file_pairs(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
    pairs = []
    for f in files:
        if "_rad" in f.lower():
            continue
        radar_candidate = f.replace(".csv", "_rad.csv")
        if radar_candidate in files:
            pairs.append((os.path.join(folder, f), os.path.join(folder, radar_candidate)))
    return pairs

# ---------------------------
# Viewer
# ---------------------------

class LidarRadarTkViewer:
    def __init__(self, folder, width=900, height=900, point_radius=2,R=None,t=None):
        self.pairs = get_file_pairs(folder)
        if not self.pairs:
            raise FileNotFoundError("No lidar/radar file pairs found in the folder.")
        self.index = 0

        self.R = R
        self.t = t
        self.scale = 50.0
        self.offset_x = width // 2
        self.offset_y = height // 2
        self.width, self.height = width, height
        self.point_radius = point_radius

        # History of selected means
        self.lidar_means = np.empty((0, 3))
        self.radar_means = np.empty((0, 3))

        # State
        self.sel_start = None
        self.sel_rect_id = None

        # Tkinter setup
        self.root = tk.Tk()
        self.root.title("Lidar-Radar Viewer (PhotoImage accelerated)")
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.imgtk = None

        # bindings
        self.root.bind("<Left>", lambda e: self.prev_file())
        self.root.bind("<Right>", lambda e: self.next_file())
        self.canvas.bind("<ButtonPress-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)

        # Initial load
        self.load_pair()
        self.root.mainloop()

    # ---------------------------
    # File loading and drawing
    # ---------------------------

    def load_pair(self):
        lidar_file, radar_file = self.pairs[self.index]
        self.lidar_points = load_points_from_csv(lidar_file)
        radar_points = load_points_from_csv(radar_file)
        self.radar_points = radar_to_lidar(radar_points,self.R,self.t)
        self.draw_points()

    def draw_points(self):
        # white background
        img = Image.new("RGB", (self.width, self.height), "white")
        pixels = img.load()
        r = self.point_radius

        def draw(points, color):
            for x, y, _ in points:
                u, v = self.world_to_screen(x, y)
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        uu, vv = u + dx, v + dy
                        if 0 <= uu < self.width and 0 <= vv < self.height:
                            pixels[uu, vv] = color

        draw(self.lidar_points, (0, 0, 255))   # blue
        draw(self.radar_points, (255, 0, 0))   # red

        # draw accumulated means
        draw(self.lidar_means, (0, 255, 0))    # green lidar means
        draw(self.radar_means, (255, 165, 0))  # orange radar means

        self.imgtk = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.imgtk)

    def world_to_screen(self, x, y):
        u = int(self.offset_x + x * self.scale)
        v = int(self.offset_y - y * self.scale)
        return u, v

    def screen_to_world(self, u, v):
        x = (u - self.offset_x) / self.scale
        y = (self.offset_y - v) / self.scale
        return x, y

    # ---------------------------
    # Events
    # ---------------------------

    def on_left_down(self, event):
        self.sel_start = (event.x, event.y)
        if self.sel_rect_id:
            self.canvas.delete(self.sel_rect_id)
        self.sel_rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red")

    def on_left_drag(self, event):
        if self.sel_start and self.sel_rect_id:
            self.canvas.coords(self.sel_rect_id, self.sel_start[0], self.sel_start[1], event.x, event.y)

    def on_left_up(self, event):
        if not self.sel_start:
            return
        x0, y0 = self.sel_start
        x1, y1 = event.x, event.y
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])

        def in_rect(points):
            uvs = np.array([self.world_to_screen(x, y) for x, y in points[:, :2]])
            mask = (uvs[:, 0] >= xmin) & (uvs[:, 0] <= xmax) & \
                   (uvs[:, 1] >= ymin) & (uvs[:, 1] <= ymax)
            return points[mask]

        sel_lidar = in_rect(self.lidar_points)
        sel_radar = in_rect(self.radar_points)

        if len(sel_lidar) > 0:
            # keep only the top 30% highest lidar points in the selected box
            z_threshold = np.percentile(sel_lidar[:, 2], 70)
            sel_lidar_top = sel_lidar[sel_lidar[:, 2] >= z_threshold]

            # fallback in case filtering becomes too aggressive
            if len(sel_lidar_top) == 0:
                sel_lidar_top = sel_lidar

            mean = sel_lidar_top.mean(axis=0)
            self.lidar_means = np.vstack([self.lidar_means, mean])
            print(f"Lidar mean: {mean}")
            print(f"Selected lidar points: {len(sel_lidar)}, top-z used: {len(sel_lidar_top)}")
        if len(sel_radar) > 0:
            mean = sel_radar.mean(axis=0)
            self.radar_means = np.vstack([self.radar_means, mean])
            print(f"Radar mean: {mean}")

        # clear selection box
        if self.sel_rect_id:
            self.canvas.delete(self.sel_rect_id)
            self.sel_rect_id = None
        self.sel_start = None

        self.draw_points()

    # ---------------------------
    # Navigation
    # ---------------------------

    def next_file(self):
        self.index = (self.index + 1) % len(self.pairs)
        self.load_pair()

    def prev_file(self):
        self.index = (self.index - 1) % len(self.pairs)
        self.load_pair()


# ---------------------------
# Run
# ---------------------------

def get_matrices_init(radar_position):
    """Get initial matrices for the viewer."""
    # original radar→lidar transform

    t_LI = np.array([0.585, 0.0, 1.859])      # lidar -> imu
    
    if radar_position == "FL":
        th = np.deg2rad(50)
        R = np.array([[np.cos(th), -np.sin(th), 0.0],
                        [np.sin(th),  np.cos(th), 0.0],
                        [0.0,         0.0,        1.0]])
        t_RI = np.array([3.441, 0.635, 0.335])   # radar -> imu
        t = t_RI - t_LI

    return R, t

if __name__ == "__main__":
    
    radar_selected = "FL"  # Change to "FR", "FC", "RL", or "RR" as needed
    FOLDER = "C:/Users/Jian/Desktop/A_Technical_Task/calib_test/test_data/test_data"

    R_ref,t_ref = get_matrices_init(radar_selected)

    viewer = LidarRadarTkViewer(FOLDER, point_radius=2,R=R_ref,t=t_ref)