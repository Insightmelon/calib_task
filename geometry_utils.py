from __future__ import annotations

from typing import Tuple

import numpy as np


def rotation_matrix_2d(theta_rad: float) -> np.ndarray:
    """Create a 2D rotation matrix for a planar yaw rotation."""
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def rotation_matrix_3d_from_yaw(theta_rad: float) -> np.ndarray:
    """Create a 3D rotation matrix for rotation around z-axis only."""
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array(
        [[c, -s, 0.0],
         [s, c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )


def apply_planar_transform_xy(
    points_xy: np.ndarray,
    theta_rad: float,
    translation_xy: np.ndarray,
) -> np.ndarray:
    """Apply a planar rigid transform in x-y."""
    rotation = rotation_matrix_2d(theta_rad)
    return (rotation @ points_xy.T).T + translation_xy


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply a 3D rigid transform to an array of points."""
    return (R @ points.T).T + t


def transform_point(point: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply a 3D rigid transform to a single point."""
    return R @ point + t


def compose_transform(
    R_ab: np.ndarray,
    t_ab: np.ndarray,
    R_bc: np.ndarray,
    t_bc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compose two transforms:
        frame_a -> frame_b
        frame_b -> frame_c
    to obtain:
        frame_a -> frame_c
    """
    R_ac = R_bc @ R_ab
    t_ac = R_bc @ t_ab + t_bc
    return R_ac, t_ac


def yaw_from_rotation_matrix(R: np.ndarray) -> float:
    """Extract yaw under the assumption that only z-rotation matters."""
    return float(np.arctan2(R[1, 0], R[0, 0]))
