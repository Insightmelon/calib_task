"""
project_radar_to_camera.py

Purpose
-------
This script performs a qualitative sanity check of the radar-to-lidar calibration
by projecting radar detections into the camera image.

Pipeline
--------
Radar points are transformed as follows:
    radar (already in lidar frame via initial extrinsics)
    -> apply residual calibration (yaw + translation refinement)
    -> lidar-to-camera extrinsics
    -> image plane projection

Visualization
-------------
The script overlays multiple elements onto the camera image to verify alignment:

1. Radar detections (blue circles with white outline)
   - All radar points are projected
   - Provides global context and shows clutter / background returns

2. Lidar correspondence (green cross)
   - Manually selected reflector position in lidar frame
   - Serves as reference (target location)

3. Refined radar prediction (yellow 'X')
   - The selected radar correspondence after residual calibration
   - Represents the predicted reflector location from radar

Interpretation
--------------
The green cross should lie on the reflector in the image. The yellow X should be
close to the green cross, especially in the horizontal direction. A vertical
offset is expected because radar elevation is unreliable and z is not optimized.


Notes
-----
- This visualization complements numerical residual analysis by providing
- an intuitive, image-level validation of calibration quality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

from geometry_utils import (
    rotation_matrix_3d_from_yaw,
    transform_point,
    transform_points,
)


# =========================
# User config: edit here only
# =========================
SCENE_ID = "31"  # one of: 8, 16, 24, 31, 64
DATA_DIR = Path("test_data/test_data")
OUTPUT_DIR = Path("results")

CALIBRATION_JSON = Path("results/calibration_result_withZ.json")
CAMERA_CALIB_NPZ = Path("cfl_calibration.npz")
LIDAR_TO_CAMERA_NPZ = Path("lidar2cfl_new_all.npz")
CORRESPONDENCES_CSV = Path("results/picked_correspondences_withZ.csv")

RADAR_CSV = DATA_DIR / f"{SCENE_ID}_rad.csv"
IMAGE_PATH = DATA_DIR / f"{SCENE_ID}.jpg"
OUTPUT_PATH = OUTPUT_DIR / f"projected_rad2img_{SCENE_ID}.jpg"


def load_radar_points(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(f"Radar CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"x", "y", "z"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    return df[["x", "y", "z"]].to_numpy(dtype=float)


def load_final_radar_to_lidar(json_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not json_path.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    R_radar_lidar = np.array(data["final_rotation_matrix_radar_to_lidar"], dtype=float)
    t_radar_lidar = np.array(data["final_translation_vector_radar_to_lidar_m"], dtype=float)
    return R_radar_lidar, t_radar_lidar


def load_residual_correction(json_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not json_path.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    theta = float(data["residual_theta_rad"])
    tx = float(data["residual_tx_m"])
    ty = float(data["residual_ty_m"])

    R = rotation_matrix_3d_from_yaw(theta)
    t = np.array([tx, ty, 0.0], dtype=float)
    return R, t


def load_camera_calibration(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Camera calibration file not found: {npz_path}")

    calib = np.load(npz_path)
    return calib["camera_matrix"], calib["dist_coeffs"]


def load_lidar_to_camera(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Lidar-to-camera calibration file not found: {npz_path}")

    calib = np.load(npz_path)
    return calib["R"], calib["t"]


def load_correspondence_for_scene(csv_path: Path, scene_id: str) -> Tuple[np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Correspondence CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {
        "scene_id",
        "radar_x",
        "radar_y",
        "radar_z",
        "lidar_x",
        "lidar_y",
        "lidar_z",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df = df.copy()
    df["scene_id"] = df["scene_id"].astype(str)
    rows = df[df["scene_id"] == str(scene_id)]
    if rows.empty:
        available = sorted(df["scene_id"].unique().tolist(), key=int)
        raise ValueError(f"Scene id '{scene_id}' not found in {csv_path}. Available scene ids: {available}")
    if len(rows) > 1:
        raise ValueError(f"Scene id '{scene_id}' appears multiple times in {csv_path}.")

    row = rows.iloc[0]
    radar_init = row[["radar_x", "radar_y", "radar_z"]].to_numpy(dtype=float)
    lidar_point = row[["lidar_x", "lidar_y", "lidar_z"]].to_numpy(dtype=float)
    return radar_init, lidar_point

def project_to_image(points_cam: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    in_front_mask = points_cam[:, 2] > 0.0
    valid_points_cam = points_cam[in_front_mask]

    if len(valid_points_cam) == 0:
        return np.empty((0, 2), dtype=float), in_front_mask

    proj_points, _ = cv2.projectPoints(
        valid_points_cam,
        np.zeros((3, 1), dtype=float),
        np.zeros((3, 1), dtype=float),
        K,
        dist_coeffs,
    )
    return proj_points.reshape(-1, 2), in_front_mask


def project_single_point(point_cam: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray | None:
    if point_cam[2] <= 0.0:
        return None

    proj_point, _ = cv2.projectPoints(
        point_cam.reshape(1, 3),
        np.zeros((3, 1), dtype=float),
        np.zeros((3, 1), dtype=float),
        K,
        dist_coeffs,
    )
    return proj_point.reshape(2)


def draw_radar_points(
    image: np.ndarray,
    proj_points: np.ndarray,
    point_radius: int = 6,
    outline_radius: int = 8,
    outline_thickness: int = 2,
) -> np.ndarray:
    output = image.copy()
    h, w = output.shape[:2]

    for u, v in proj_points:
        u_i = int(round(float(u)))
        v_i = int(round(float(v)))
        if 0 <= u_i < w and 0 <= v_i < h:
            cv2.circle(output, (u_i, v_i), outline_radius, (255, 255, 255), outline_thickness, cv2.LINE_AA)
            cv2.circle(output, (u_i, v_i), point_radius, (255, 0, 0), -1, cv2.LINE_AA)
    return output

def draw_marker(
    image: np.ndarray,
    point_uv: np.ndarray | None,
    color: Tuple[int, int, int],
    marker_type: str,
    size: int = 14,
    thickness: int = 2,
) -> np.ndarray:
    if point_uv is None:
        return image

    output = image.copy()
    h, w = output.shape[:2]
    u_i = int(round(float(point_uv[0])))
    v_i = int(round(float(point_uv[1])))

    if not (0 <= u_i < w and 0 <= v_i < h):
        return output

    if marker_type == "cross":
        cv2.line(output, (u_i - size, v_i), (u_i + size, v_i), color, thickness, cv2.LINE_AA)
        cv2.line(output, (u_i, v_i - size), (u_i, v_i + size), color, thickness, cv2.LINE_AA)
    elif marker_type == "x":
        cv2.line(output, (u_i - size, v_i - size), (u_i + size, v_i + size), color, thickness, cv2.LINE_AA)
        cv2.line(output, (u_i - size, v_i + size), (u_i + size, v_i - size), color, thickness, cv2.LINE_AA)
    else:
        raise ValueError(f"Unsupported marker_type: {marker_type}")

    return output

def draw_legend_box(image: np.ndarray) -> np.ndarray:
    legend_items = [
        ("radar points", (255, 0, 0)),          # blue
        ("lidar correspondence", (0, 255, 0)),  # green
        ("refined radar correspondence", (0, 255, 255)),  # yellow
    ]

    output = image.copy()
    overlay = output.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = output.shape[:2]

    font_scale = max(0.8, min(1.1, w / 1600.0))
    text_thickness = 2
    padding = max(14, int(round(w / 120.0)))
    margin = max(12, int(round(w / 140.0)))

    text_sizes = [
        cv2.getTextSize(text, font, font_scale, text_thickness)[0]
        for text, _ in legend_items
    ]
    max_text_width = max(size[0] for size in text_sizes)
    max_text_height = max(size[1] for size in text_sizes)

    line_gap = max(10, int(round(max_text_height * 0.8)))
    line_step = max_text_height + line_gap

    box_width = max_text_width + 2 * padding
    box_height = len(legend_items) * line_step + 2 * padding - line_gap

    x1 = w - box_width - margin
    y1 = margin
    x2 = w - margin
    y2 = y1 + box_height

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (220, 220, 220), -1) #background color
    output = cv2.addWeighted(overlay, 0.9, output, 0.1, 0.0)
    cv2.rectangle(output, (x1, y1), (x2, y2), (120, 120, 120), 1, cv2.LINE_AA)

    baseline_y = y1 + padding + max_text_height

    for idx, (text, color) in enumerate(legend_items):
        y = baseline_y + idx * line_step
        cv2.putText(
            output,
            text,
            (x1 + padding, y),
            font,
            font_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )

    return output

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    radar_points = load_radar_points(RADAR_CSV)
    R_radar_lidar, t_radar_lidar = load_final_radar_to_lidar(CALIBRATION_JSON)
    R_residual, t_residual = load_residual_correction(CALIBRATION_JSON)
    K, dist_coeffs = load_camera_calibration(CAMERA_CALIB_NPZ)
    R_lidar_cam, t_lidar_cam = load_lidar_to_camera(LIDAR_TO_CAMERA_NPZ)

    radar_init_point_lidar, lidar_point_lidar = load_correspondence_for_scene(CORRESPONDENCES_CSV, SCENE_ID)

    radar_points_lidar = transform_points(radar_points, R_radar_lidar, t_radar_lidar)
    radar_points_cam = transform_points(radar_points_lidar, R_lidar_cam, t_lidar_cam)
    proj_points, in_front_mask = project_to_image(radar_points_cam, K, dist_coeffs)

    refined_radar_point_lidar = transform_point(radar_init_point_lidar, R_residual, t_residual)
    refined_radar_point_cam = transform_point(refined_radar_point_lidar, R_lidar_cam, t_lidar_cam)
    lidar_point_cam = transform_point(lidar_point_lidar, R_lidar_cam, t_lidar_cam)

    refined_radar_uv = project_single_point(refined_radar_point_cam, K, dist_coeffs)
    lidar_uv = project_single_point(lidar_point_cam, K, dist_coeffs)

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    output_image = draw_radar_points(image, proj_points)
    output_image = draw_marker(output_image, lidar_uv, color=(0, 255, 0), marker_type="cross", size=14, thickness=2)
    output_image = draw_marker(output_image, refined_radar_uv, color=(0, 255, 255), marker_type="x", size=14, thickness=2)
    output_image = draw_legend_box(output_image)

    cv2.imwrite(str(OUTPUT_PATH), output_image)

    print(f"Saved projection image to: {OUTPUT_PATH}")
    print(f"Scene id: {SCENE_ID}")
    print(f"Radar CSV: {RADAR_CSV}")
    print(f"Image path: {IMAGE_PATH}")
    print(f"Radar points total: {len(radar_points)}")
    print(f"Radar points in front of camera: {int(np.sum(in_front_mask))}")
    print(f"Projected radar image points: {len(proj_points)}")
    print(f"Lidar correspondence (lidar frame): {lidar_point_lidar.tolist()}")
    print(f"Refined radar prediction (lidar frame): {refined_radar_point_lidar.tolist()}")


if __name__ == "__main__":
    main()
