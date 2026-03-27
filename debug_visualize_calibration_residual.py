"""
debug_visualize_calibration_residual.py

Visualize radar/lidar correspondences before and after residual refinement.

Important:
    The radar points in picked_correspondences_withZ.csv are already projected into
    the lidar frame using the initial calibration from quick_view.py.

Therefore:
    - "before" means: pre-aligned radar points from quick_view.py
    - "after" means: the same points after the estimated residual correction

This script does not compare raw radar-frame points to lidar points.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV = Path("results/picked_correspondences_withZ.csv")
INPUT_JSON = Path("results/calibration_result_withZ.json")
OUTPUT_PLOT = Path("results/calibration_before_after_refinement.jpg")


def rotation_matrix_2d(theta_rad: float) -> np.ndarray:
    """
    Create a 2D rotation matrix.

    Args:
        theta_rad: Rotation angle in radians.

    Returns:
        A 2x2 rotation matrix.
    """
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)


def apply_residual_correction(
    radar_init_xy: np.ndarray,
    residual_theta_rad: float,
    residual_translation_xy: np.ndarray,
) -> np.ndarray:
    """
    Apply the residual planar correction to pre-aligned radar points.

    Args:
        radar_init_xy: Radar points already in lidar frame after initial alignment.
        residual_theta_rad: Residual yaw correction in radians.
        residual_translation_xy: Residual translation [dtx, dty].

    Returns:
        Refined radar points in lidar frame.
    """
    rotation = rotation_matrix_2d(residual_theta_rad)
    return (rotation @ radar_init_xy.T).T + residual_translation_xy


def load_correspondences(csv_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Load selected correspondences.

    Returns:
        scene_ids: Scene identifiers used for annotation
        radar_init_xy: Pre-aligned radar points in lidar frame
        lidar_xy: Lidar correspondence points
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing correspondence CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"scene_id", "radar_x", "radar_y", "lidar_x", "lidar_y"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df = df.sort_values("scene_id", kind="stable")

    scene_ids = df["scene_id"].astype(str).tolist()
    radar_init_xy = df[["radar_x", "radar_y"]].to_numpy(dtype=float)
    lidar_xy = df[["lidar_x", "lidar_y"]].to_numpy(dtype=float)
    return scene_ids, radar_init_xy, lidar_xy


def load_result(json_path: Path) -> Tuple[float, float, float]:
    """
    Load residual correction parameters from calibration_result.json.

    Returns:
        residual_theta_rad, residual_tx_m, residual_ty_m
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Missing calibration result JSON: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    return (
        float(data["residual_theta_rad"]),
        float(data["residual_tx_m"]),
        float(data["residual_ty_m"]),
    )


def annotate_points(points_xy: np.ndarray, scene_ids: List[str]) -> None:
    """
    Add scene-id labels to the plot.
    """
    for scene_id, (x, y) in zip(scene_ids, points_xy):
        plt.annotate(
            scene_id,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )


def plot_before_after_refinement(
    scene_ids: List[str],
    radar_before_xy: np.ndarray,
    lidar_xy: np.ndarray,
    radar_after_xy: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot pre-aligned radar points, refined radar points, and lidar correspondences.
    """
    plt.figure(figsize=(9, 7))

    plt.scatter(lidar_xy[:, 0], lidar_xy[:, 1], marker="o", label="Lidar correspondence")
    plt.scatter(radar_before_xy[:, 0], radar_before_xy[:, 1], marker="x", label="Radar (before refinement)")
    plt.scatter(radar_after_xy[:, 0], radar_after_xy[:, 1], marker="+", label="Radar (after refinement)")

    for idx in range(len(lidar_xy)):
        plt.plot(
            [radar_before_xy[idx, 0], lidar_xy[idx, 0]],
            [radar_before_xy[idx, 1], lidar_xy[idx, 1]],
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
        plt.plot(
            [radar_after_xy[idx, 0], lidar_xy[idx, 0]],
            [radar_after_xy[idx, 1], lidar_xy[idx, 1]],
            linestyle="-",
            linewidth=1,
            alpha=0.7,
        )

    annotate_points(lidar_xy, scene_ids)
    annotate_points(radar_before_xy, scene_ids)
    annotate_points(radar_after_xy, scene_ids)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Radar-to-Lidar Refinement: Before vs After Residual Correction")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()


def main() -> None:
    """Run the visualization pipeline."""
    scene_ids, radar_before_xy, lidar_xy = load_correspondences(INPUT_CSV)
    residual_theta_rad, residual_tx_m, residual_ty_m = load_result(INPUT_JSON)

    radar_after_xy = apply_residual_correction(
        radar_init_xy=radar_before_xy,
        residual_theta_rad=residual_theta_rad,
        residual_translation_xy=np.array([residual_tx_m, residual_ty_m], dtype=float),
    )

    plot_before_after_refinement(
        scene_ids=scene_ids,
        radar_before_xy=radar_before_xy,
        lidar_xy=lidar_xy,
        radar_after_xy=radar_after_xy,
        output_path=OUTPUT_PLOT,
    )


if __name__ == "__main__":
    main()
