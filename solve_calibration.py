"""
solve_calibration.py

Estimate radar-to-lidar extrinsics from manually selected correspondences.

Important note about the input CSV:
    The radar points in picked_correspondences_withZ.csv were selected in quick_view.py.
    In that viewer, radar points are already projected into the lidar frame using
    the provided initial calibration.

Therefore, this script:
1) estimates a residual planar correction on top of the initial calibration
2) composes the residual correction with the initial calibration
3) outputs the final absolute radar-to-lidar extrinsics

Task assumptions from README:
- radar elevation is unreliable -> optimize only yaw, tx, ty
- tz is kept fixed from the provided initial guess
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from geometry_utils import (
    apply_planar_transform_xy,
    compose_transform,
    rotation_matrix_3d_from_yaw,
    yaw_from_rotation_matrix,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_CSV = Path("results/picked_correspondences_withZ.csv")
OUTPUT_JSON = Path("results/calibration_result_withZ.json")

# Optional debug output:
# Save per-scene residuals before/after refinement for quick numeric inspection.
# Not required for the main workflow or for camera projection sanity check.
SAVE_RESIDUALS_CSV = True
OUTPUT_RESIDUALS_CSV = Path("results/residuals_per_point_withZ.csv")

# Initial guess from the README / quick_view.py
INITIAL_THETA_DEG = 50.0
INITIAL_TX_M = 2.856
INITIAL_TY_M = 0.635
FIXED_TZ_M = -1.524

ROBUST_LOSS = "soft_l1"
ROBUST_F_SCALE_M = 0.10


# ---------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Container for solver results."""
    success: bool
    message: str

    num_correspondences: int
    scene_ids: list[str]

    # Initial absolute extrinsics
    initial_theta_deg: float
    initial_theta_rad: float
    initial_tx_m: float
    initial_ty_m: float
    initial_tz_m: float

    # Residual correction estimated from the pre-aligned CSV correspondences
    residual_theta_deg: float
    residual_theta_rad: float
    residual_tx_m: float
    residual_ty_m: float

    # Final absolute extrinsics after composition
    final_theta_deg: float
    final_theta_rad: float
    final_tx_m: float
    final_ty_m: float
    final_tz_m: float

    # Fit quality
    rmse_xy_m: float
    mean_abs_error_x_m: float
    mean_abs_error_y_m: float
    max_point_error_xy_m: float

    # Solver configuration
    robust_loss: str
    robust_f_scale_m: float


# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------

def load_correspondences(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load selected correspondences.

    Expected columns:
        radar_x, radar_y, lidar_x, lidar_y

    Important:
        radar_x / radar_y are already pre-aligned into lidar frame by quick_view.py
        using the initial calibration.

    Args:
        csv_path: Input CSV path.

    Returns:
        radar_init_xy: Pre-aligned radar points in lidar frame, shape (N, 2)
        lidar_xy: Lidar correspondence points in lidar frame, shape (N, 2)
        df: Original DataFrame
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Correspondence file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"scene_id", "radar_x", "radar_y", "lidar_x", "lidar_y"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df = df.copy()
    df["scene_id"] = df["scene_id"].astype(str)
    df = df.sort_values("scene_id", key=lambda s: s.astype(int)).reset_index(drop=True)

    radar_init_xy = df[["radar_x", "radar_y"]].to_numpy(dtype=float)
    lidar_xy = df[["lidar_x", "lidar_y"]].to_numpy(dtype=float)

    if len(radar_init_xy) < 2:
        raise ValueError("At least two correspondences are required.")

    return radar_init_xy, lidar_xy, df


# ---------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------

def residual_vector(
    params: np.ndarray,
    radar_init_xy: np.ndarray,
    lidar_xy: np.ndarray,
) -> np.ndarray:
    """
    Residual vector for least-squares optimization.

    The optimization is performed on the pre-aligned radar points, so the
    variables here represent a residual planar correction:
        [dtheta, dtx, dty]

    Args:
        params: Residual parameters [dtheta_rad, dtx_m, dty_m].
        radar_init_xy: Pre-aligned radar points in lidar frame, shape (N, 2).
        lidar_xy: Target lidar points in lidar frame, shape (N, 2).

    Returns:
        Flattened residual vector of shape (2N,).
    """
    dtheta_rad, dtx_m, dty_m = params
    predicted_xy = apply_planar_transform_xy(
        radar_init_xy,
        dtheta_rad,
        np.array([dtx_m, dty_m], dtype=float),
    )
    return (predicted_xy - lidar_xy).ravel()


def solve_calibration(
    radar_init_xy: np.ndarray,
    lidar_xy: np.ndarray,
    scene_ids: list[str],
) -> Tuple[CalibrationResult, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve for the residual correction and compose it with the initial transform
    to obtain final absolute radar-to-lidar extrinsics.

    Args:
        radar_init_xy: Pre-aligned radar points in lidar frame, shape (N, 2).
        lidar_xy: Lidar correspondence points, shape (N, 2).
        scene_ids: scene ids

    Returns:
        result: CalibrationResult
        refined_radar_xy: Radar points after residual refinement, shape (N, 2)
        R_final: Final absolute 3x3 radar-to-lidar rotation matrix
        t_final: Final absolute 3D radar-to-lidar translation vector
    """
    initial_params = np.array([0.0, 0.0, 0.0], dtype=float)

    optimization = least_squares(
        fun=residual_vector,
        x0=initial_params,
        args=(radar_init_xy, lidar_xy),
        loss=ROBUST_LOSS,
        f_scale=ROBUST_F_SCALE_M,
    )

    dtheta_rad, dtx_m, dty_m = optimization.x

    refined_radar_xy = apply_planar_transform_xy(
        radar_init_xy,
        dtheta_rad,
        np.array([dtx_m, dty_m], dtype=float),
    )

    errors_xy = refined_radar_xy - lidar_xy
    error_norms_xy = np.linalg.norm(errors_xy, axis=1)

    rmse_xy_m = float(np.sqrt(np.mean(np.sum(errors_xy**2, axis=1))))
    mean_abs_error_x_m = float(np.mean(np.abs(errors_xy[:, 0])))
    mean_abs_error_y_m = float(np.mean(np.abs(errors_xy[:, 1])))
    max_point_error_xy_m = float(np.max(error_norms_xy))

    # Initial absolute transform (radar -> lidar)
    theta0_rad = np.deg2rad(INITIAL_THETA_DEG)
    R0 = rotation_matrix_3d_from_yaw(theta0_rad)
    t0 = np.array([INITIAL_TX_M, INITIAL_TY_M, FIXED_TZ_M], dtype=float)

    # Residual transform, also interpreted as radar_init -> lidar
    Rd = rotation_matrix_3d_from_yaw(dtheta_rad)
    td = np.array([dtx_m, dty_m, 0.0], dtype=float)

    # Final absolute transform
    R_final, t_final = compose_transform(R0, t0, Rd, td)
    theta_final_rad = yaw_from_rotation_matrix(R_final)
    theta_final_deg = float(np.rad2deg(theta_final_rad))

    result = CalibrationResult(
        success=bool(optimization.success),
        message=str(optimization.message),
        num_correspondences=int(len(radar_init_xy)),
        scene_ids=scene_ids,
        initial_theta_deg=float(INITIAL_THETA_DEG),
        initial_theta_rad=float(theta0_rad),
        initial_tx_m=float(INITIAL_TX_M),
        initial_ty_m=float(INITIAL_TY_M),
        initial_tz_m=float(FIXED_TZ_M),
        residual_theta_deg=float(np.rad2deg(dtheta_rad)),
        residual_theta_rad=float(dtheta_rad),
        residual_tx_m=float(dtx_m),
        residual_ty_m=float(dty_m),
        final_theta_deg=theta_final_deg,
        final_theta_rad=float(theta_final_rad),
        final_tx_m=float(t_final[0]),
        final_ty_m=float(t_final[1]),
        final_tz_m=float(t_final[2]),
        rmse_xy_m=rmse_xy_m,
        mean_abs_error_x_m=mean_abs_error_x_m,
        mean_abs_error_y_m=mean_abs_error_y_m,
        max_point_error_xy_m=max_point_error_xy_m,
        robust_loss=ROBUST_LOSS,
        robust_f_scale_m=float(ROBUST_F_SCALE_M),
    )

    return result, refined_radar_xy, R_final, t_final


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_calibration_result(
    result: CalibrationResult,
    R_final: np.ndarray,
    t_final: np.ndarray,
    output_path: Path,
) -> None:
    payload = asdict(result)

    # Highlight the key deliverables for downstream scripts and for quick reading.
    payload["key_results"] = {
        "final_theta_deg": result.final_theta_deg,
        "final_translation_xyz_m": [result.final_tx_m, result.final_ty_m, result.final_tz_m],
    }
    payload["final_rotation_matrix_radar_to_lidar"] = R_final.tolist()
    payload["final_translation_vector_radar_to_lidar_m"] = t_final.tolist()

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_residual_report(
    df: pd.DataFrame,
    refined_radar_xy: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save per-point refined coordinates and residuals.

    Args:
        df: Original correspondence DataFrame.
        refined_radar_xy: Refined radar points in lidar frame, shape (N, 2).
        output_path: Output CSV path.
    """
    out_df = df.copy()
    out_df["refined_radar_x"] = refined_radar_xy[:, 0]
    out_df["refined_radar_y"] = refined_radar_xy[:, 1]
    out_df["residual_x_m"] = out_df["refined_radar_x"] - out_df["lidar_x"]
    out_df["residual_y_m"] = out_df["refined_radar_y"] - out_df["lidar_y"]
    out_df["residual_norm_xy_m"] = np.sqrt(
        out_df["residual_x_m"] ** 2 + out_df["residual_y_m"] ** 2
    )
    out_df.to_csv(output_path, index=False)


def print_summary(result: CalibrationResult) -> None:
    """
    Print a concise summary to the console.

    Args:
        result: Solver result.
    """
    print("\n=== Radar-to-Lidar Calibration Result ===")
    print(f"Success:                    {result.success}")
    print(f"Message:                    {result.message}")
    print(f"Scene ids:                  {result.scene_ids}")
    print(f"Number of correspondences:  {result.num_correspondences}")
    print(f"Initial theta [deg]:        {result.initial_theta_deg:.6f}")
    print(f"Residual theta [deg]:       {result.residual_theta_deg:.6f}")
    print(f"Residual translation xy [m]: [{result.residual_tx_m:.6f}, {result.residual_ty_m:.6f}]")

    print(f"Final theta [deg]:          {result.final_theta_deg:.6f}")
    print(
        "Final translation xyz [m]: "
        f"[{result.final_tx_m:.6f}, {result.final_ty_m:.6f}, {result.final_tz_m:.6f}]"
    )

    print(f"RMSE XY [m]:                {result.rmse_xy_m:.6f}")
    print(f"Mean |error_x| [m]:         {result.mean_abs_error_x_m:.6f}")
    print(f"Mean |error_y| [m]:         {result.mean_abs_error_y_m:.6f}")
    print(f"Max point error XY [m]:     {result.max_point_error_xy_m:.6f}")


def main() -> None:
    """Run the calibration pipeline."""
    radar_init_xy, lidar_xy, df = load_correspondences(INPUT_CSV)
    scene_ids = df["scene_id"].astype(str).tolist()
    result, refined_radar_xy, R_final, t_final = solve_calibration(radar_init_xy, lidar_xy, scene_ids)

    save_calibration_result(result, R_final, t_final, OUTPUT_JSON)
    if SAVE_RESIDUALS_CSV:
        save_residual_report(df, refined_radar_xy, OUTPUT_RESIDUALS_CSV)
    print_summary(result)


if __name__ == "__main__":
    main()
