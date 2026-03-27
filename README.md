# Radar-Lidar Calibration Task Solution

This repository contains my solution to a radar-lidar extrinsic calibration task. The goal is to estimate the radar-to-lidar extrinsics from a small set of scenes containing a corner reflector, using the provided initial guess and validating the result through camera reprojection.

My workflow is:

1. select one radar/lidar corner-reflector correspondence per scene with a lightweight viewer
2. solve a planar residual calibration on top of the provided initial extrinsics
3. validate the result numerically and by projecting radar points into the camera image

The original task description is preserved in [TASK_README.md](TASK_README.md).

## Approach

### 1. Correspondence selection

[quick_view.py](quick_view.py) is based on the viewer script provided with the task and is used to browse the lidar/radar pairs scene by scene and manually select the corner reflector.

- I extended the provided viewer rather than replacing it with a new tool.
- Radar points are first transformed into the lidar frame using the provided initial calibration.
- A selection box is drawn around the reflector region.
- For lidar, I keep only the top 40% highest points in the selected box before computing the mean point. This helps bias the picked lidar point toward the reflector return instead of ground or nearby clutter.
- For radar, I compute the mean of the selected radar points in the same box.
- One correspondence is extracted per scene and saved automatically to `results/picked_correspondences_withZ.csv`.
- If the same `scene_id` already exists, the script asks whether to overwrite it.
- I also added automatic creation of the `results/` folder when needed.

### 2. Calibration solve

[solve_calibration.py](solve_calibration.py) reads the picked correspondences and estimates a residual planar correction.

- The radar correspondences stored in the CSV are already pre-aligned into the lidar frame using the initial guess from `quick_view.py`.
- I therefore solve for a residual 2D rigid transform in the ground plane.
- Since radar elevation is unreliable, I optimize only `yaw`, `tx`, and `ty`, while keeping `tz` fixed from the provided initial calibration.
- I use `scipy.optimize.least_squares` with a robust loss to reduce the influence of imperfect manual picks and measurement noise.
- The residual correction is then composed with the initial transform to produce the final absolute radar-to-lidar extrinsics.

The main output is `results/calibration_result_withZ.json`.

An optional per-point residual report can also be generated for quick numeric inspection, but it is not required for the main workflow.

### 3. Camera projection validation

[project_radar_to_camera.py](project_radar_to_camera.py) performs a qualitative validation by projecting radar detections into the camera image.

- Radar points are transformed using the estimated radar-to-lidar extrinsics and the provided lidar-to-camera extrinsics.
- The manually selected lidar correspondence and the refined radar correspondence are also projected into the image.
- The script saves an output image per scene in `results/`, for example `projected_rad2img_16.jpg`.

What I check in the projected image:

- the lidar correspondence should align with the reflector in the image
- the refined radar correspondence should lie close to it, especially horizontally
- some vertical offset is expected because radar z is unreliable and was not optimized

### 4. Optional debug visualization

[debug_visualize_calibration_residual.py](debug_visualize_calibration_residual.py) is an optional debug script. It plots radar points before refinement, radar points after refinement, and lidar correspondences in the XY plane to show whether the residual calibration reduces the mismatch.

This script is not required for the main workflow, but it is useful for a quick sanity check without going through camera projection.

## Repository Structure

- `quick_view.py`: manually pick radar/lidar correspondences
- `solve_calibration.py`: estimate radar-to-lidar calibration from picked correspondences
- `project_radar_to_camera.py`: validate calibration by reprojection into the camera image
- `debug_visualize_calibration_residual.py`: optional 2D debug visualization
- `geometry_utils.py`: shared rigid-transform and rotation helpers
- `results/`: generated correspondences, calibration outputs, debug plots, and projected images

## Installation

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Note: `quick_view.py` uses `tkinter`, which should be available in a standard local Python installation with Tk support.

## Main Workflow

Run the scripts in this order:

```bash
python quick_view.py
python solve_calibration.py
python project_radar_to_camera.py
```

Optional debug step:

```bash
python debug_visualize_calibration_residual.py
```

## Outputs

The scripts generate outputs under `results/`, including:

- `picked_correspondences_withZ.csv`: one picked radar/lidar correspondence per scene
- `calibration_result_withZ.json`: estimated final radar-to-lidar extrinsics
- `projected_rad2img_<scene>.jpg`: camera reprojection validation images
- `calibration_before_after_refinement.jpg`: optional XY-plane debug plot
- `residuals_per_point_withZ.csv`: optional per-point residual report

I also kept screenshot images of the selected regions for each scene in `results/` for convenience when reviewing the picked correspondences without reopening the viewer.

## Notes

- This solution intentionally favors a simple, task-focused workflow over a heavier fully automatic registration pipeline.
- I did not use ICP because the task can be solved reliably from sparse manually selected corner-reflector correspondences, and the README explicitly suggests a quick correspondence-based approach.
- The calibration model is intentionally constrained to the ground plane because radar elevation is unreliable in this dataset.

