#!/usr/bin/env bash
# ISL UMI setup script
# Clones the ISL-patched UMI repo and verifies the conda environment.
set -e

REPO_URL="https://github.com/profshrestha/universal_manipulation_interface.git"
BRANCH="isl-rx150-fixes"
INSTALL_DIR="${1:-$HOME/umi-isl}"

echo "=== ISL UMI Setup ==="
echo "Install dir : $INSTALL_DIR"
echo "Branch      : $BRANCH"
echo ""

# ── 1. Clone or update ──────────────────────────────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "[1/3] Repo already exists — pulling latest..."
    git -C "$INSTALL_DIR" pull --ff-only
else
    echo "[1/3] Cloning repo..."
    git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi

# ── 2. Conda environment ─────────────────────────────────────────────────────
if conda env list 2>/dev/null | grep -q "^umi "; then
    echo "[2/3] conda env 'umi' already exists — skipping creation."
else
    echo "[2/3] Creating conda env 'umi' (this takes a few minutes)..."
    conda env create -n umi -f "$INSTALL_DIR/conda_environment.yaml"
fi

# ── 3. Verify OpenCV ArUco API ───────────────────────────────────────────────
echo "[3/3] Verifying OpenCV..."
OPENCV_VER=$(conda run -n umi python -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "ERROR")
if [ "$OPENCV_VER" = "ERROR" ]; then
    echo "  ERROR: could not import cv2 in the 'umi' env. Check your conda install."
    exit 1
fi
echo "  OpenCV $OPENCV_VER"

conda run -n umi python -c "import cv2; cv2.aruco.ArucoDetector" 2>/dev/null \
    && echo "  ArucoDetector API: OK" \
    || { echo "  ERROR: ArucoDetector not available (OpenCV < 4.7). Reinstall env from conda_environment.yaml."; exit 1; }

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Run the SLAM pipeline:"
echo "  conda run -n umi python $INSTALL_DIR/run_slam_pipeline.py <session_dir>"
echo ""
echo "Run training after SLAM:"
echo "  conda run -n umi python $INSTALL_DIR/run_training.py <path/to/replay_buffer.zarr.zip>"
