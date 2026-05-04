# ISL UMI — Student Setup

This is the ISL-patched fork of [Universal Manipulation Interface](https://github.com/real-stanford/universal_manipulation_interface).
It fixes compatibility issues with OpenCV 4.8+ and adds support for symlinked sessions and the RX150 gripper with 16 mm ArUco tags.

## Quick Start

### 1 — Run the setup script

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/profshrestha/universal_manipulation_interface/isl-rx150-fixes/setup_isl.sh)
```

This will:
- Clone the repo to `~/umi-isl/`
- Create the `umi` conda environment (skipped if it already exists)
- Verify your OpenCV version supports the new ArUco API

### 2 — Organize your session

Your raw videos must be in a directory with this structure:

```
session_name/
└── raw_videos/
    ├── mapping.mp4          ← longest video, used for map building
    ├── gripper_calibration/ ← one clip per camera showing open/close
    │   ├── cam_left.mp4
    │   └── cam_right.mp4
    ├── demo_001_left.mp4
    ├── demo_001_right.mp4
    └── ...
```

> If `mapping.mp4` or `gripper_calibration/` are missing, step 00 creates them automatically
> (largest file → mapping, earliest clip per camera → gripper calibration).

### 3 — Run the full pipeline

```bash
conda run -n umi python ~/umi-isl/run_slam_pipeline.py <path/to/session_name>
```

**For RX150 with 16 mm ArUco tags** (default — no extra flags needed):
```bash
conda run -n umi python ~/umi-isl/run_slam_pipeline.py ~/my_data/session_001/
```

**For standard UMI gripper with 22 mm tags:**
```bash
conda run -n umi python ~/umi-isl/run_slam_pipeline.py --nominal_z 0.072 ~/my_data/session_001/
```

Output: `session_name/replay_buffer.zarr.zip`

### 4 — Train

```bash
conda run -n umi python ~/umi-isl/run_training.py session_name/replay_buffer.zarr.zip
```

Checkpoints saved to `~/umi_checkpoints/<run_name>/`.

---

## What changed from upstream

| File | Change |
|------|--------|
| `umi/common/cv_util.py` | Updated ArUco API for OpenCV ≥ 4.8; added guard against degenerate corner detections |
| `scripts_slam_pipeline/00_process_videos.py` | Handles symlinked session directories (hard-links target instead of skipping) |
| `scripts_slam_pipeline/04_detect_aruco.py` | Surfaces subprocess failures with stderr output instead of silent discard |
| `scripts_slam_pipeline/05_run_calibrations.py` | Added `--nominal_z` passthrough to `calibrate_gripper_range.py` |
| `scripts_slam_pipeline/06_generate_dataset_plan.py` | NaN gripper calibration fallback; deterministic glob ordering |
| `run_slam_pipeline.py` | Added `--nominal_z` (default 0.057 for RX150/16 mm), `--output`, and step 07 |
| `diffusion_policy/model/common/lr_scheduler.py` | Fixed broken import from newer `diffusers` |
| `diffusion_policy/workspace/train_diffusion_unet_image_workspace.py` | Made W&B optional via `WANDB_MODE` env var |
| `diffusion_policy/config/task/umi_bimanual.yaml` | Fixed Hydra dataset path override |

---

## Troubleshooting

**`step 01 found 0 video dirs`**
Your `raw_videos/` only has symlinks. This fork handles that — make sure you cloned this fork, not upstream.

**`ArUco detection failed` / `tag_detection.pkl` missing**
Step 04 prints failures with stderr. Check for `(-215:Assertion failed) m.dims >= 2` — this is the OpenCV 4.11 solvePnP bug, fixed in this fork. Verify you're using the patched `cv_util.py`.

**`IndexError` in step 06 / NaN gripper widths**
One camera's gripper calibration clip had no valid ArUco detections. This fork falls back to the other gripper's calibration automatically. Check that tags were visible in your calibration clip.

**OOM during training**
Reduce batch size: `run_training.py replay_buffer.zarr.zip -bs 16`
