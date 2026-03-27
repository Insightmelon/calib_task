# Radar-Lidar Calibration Task Solution

This repository contains my solution to a radar-lidar extrinsic calibration task. The goal is to estimate the radar-to-lidar extrinsics from a small set of scenes containing a corner reflector, starting from the provided initial guess and validating the result through camera reprojection.

The workflow is:

1. select one radar/lidar corner-reflector correspondence per scene with a lightweight viewer
2. solve a planar residual calibration on top of the provided initial extrinsics
3. validate the result by projecting radar points into the camera image

The original task description is preserved in [TASK_README.md](TASK_README.md).

## Assumptions and Design Choices

- The radar correspondences stored in the CSV are already pre-aligned into the lidar frame using the provided initial calibration.
- Radar elevation is unreliable in this dataset, so the calibration is modeled as a planar rigid transform and only `yaw`, `tx`, and `ty` are optimized.
- The vertical translation `tz` is kept fixed from the provided initial calibration.
- The solve is formulated as a residual correction on top of the provided initial extrinsics, not as a full 6-DoF re-estimation.
- A robust least-squares loss is used to reduce sensitivity to imperfect manual picks and measurement noise.
- The implementation intentionally stays lightweight and task-focused rather than introducing a heavier fully automatic registration pipeline.

### 1. Correspondence selection

[quick_view.py](quick_view.py) is based on the viewer script provided with the task and is used to browse lidar/radar pairs scene by scene and manually select the corner reflector.

- Radar points are first transformed into the lidar frame using the provided initial calibration.
- A selection box is drawn around the reflector region.
- Compared with the provided script, I added a height-based lidar selection that keeps only the top 40% highest lidar points inside the selected box before computing the mean. This helps bias the picked lidar point toward the reflector return instead of ground or nearby clutter.
- I also added automatic saving of one picked radar/lidar correspondence per scene to `results/picked_correspondences_withZ.csv`, duplicate `scene_id` checking with overwrite confirmation, and automatic creation of the `results/` folder when needed.

### 2. Calibration solve

[solve_calibration.py](solve_calibration.py) reads the picked correspondences and estimates a residual planar correction.

- I therefore solve for a residual 2D rigid transform in the ground plane.
- I use `scipy.optimize.least_squares` with a robust loss to reduce the influence of imperfect manual picks and measurement noise.
- The residual correction is then composed with the initial transform to produce the final absolute radar-to-lidar extrinsics.

The main output is `results/calibration_result_withZ.json`. An optional per-point residual report can also be generated for quick numeric inspection, but it is not required for the main workflow.

### 3. Camera projection validation

[project_radar_to_camera.py](project_radar_to_camera.py) performs a qualitative validation by projecting radar detections into the camera image.

- Radar points are transformed with the estimated radar-to-lidar extrinsics and the provided lidar-to-camera extrinsics.
- All radar detections are projected for global context.
- The manually selected lidar correspondence is projected as a reference for where the reflector should appear in the image.
- The refined radar correspondence is also projected to check whether it lands close to the lidar reference after calibration.
- The script saves an output image per scene in `results/`, for example `projected_rad2img_16.jpg`.

In the projected image, I check that:

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

## Example Result

Using the five selected scene correspondences currently stored in `results/picked_correspondences_withZ.csv`, the calibration result is:

- Final yaw: `44.62 deg`
- Final translation: `[2.656, 0.450, -1.524] m`
- RMSE (XY): `0.105 m`

The full result is saved in `results/calibration_result_withZ.json`.

## Outputs

The scripts generate outputs under `results/`, including:

- `picked_correspondences_withZ.csv`: one picked radar/lidar correspondence per scene
- `calibration_result_withZ.json`: estimated final radar-to-lidar extrinsics
- `projected_rad2img_<scene>.jpg`: camera reprojection validation images
- `calibration_before_after_refinement.jpg`: optional XY-plane debug plot
- `residuals_per_point_withZ.csv`: optional per-point residual report

I also kept screenshot images of the selected regions for each scene in `results/` for convenience when reviewing the picked correspondences without reopening the viewer.

## Limitations

- Only one manually selected radar/lidar correspondence is used per scene.
- The final result depends on the quality of the manual picks.
- Radar elevation is not optimized, so some vertical mismatch in the image projection is expected.
- Camera projection is used as a qualitative sanity check rather than a direct optimization objective.
- I did not use ICP because the task can be solved reliably from sparse manually selected corner-reflector correspondences, and the original task description explicitly suggests a quick correspondence-based approach.

