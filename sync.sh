#!/usr/bin/env bash
# sync.sh — Pull latest code from GitHub and deploy the plugin into an AlpaSim checkout.
#
# Usage:
#   ./sync.sh                                   # deploys into $ALPASIM_ROOT (default /home/harun/alpasim)
#   ALPASIM_ROOT=$PWD/third_party/alpasim ./sync.sh   # deploy into the pinned submodule instead

set -euo pipefail

STUDENT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALPASIM_ROOT="${ALPASIM_ROOT:-/home/harun/alpasim}"

echo "[sync] Pushing and pulling from GitHub..."
cd "$STUDENT_REPO"
git push
git pull --ff-only
echo "[sync] Simulator pin: $(git ls-tree HEAD third_party/alpasim | awk '{print $3}')"

echo "[sync] Deploying plugin into $ALPASIM_ROOT/plugins/alpasim_student ..."
mkdir -p "$ALPASIM_ROOT/plugins/alpasim_student"
rsync -a --delete --delete-excluded --exclude __pycache__ \
  "$STUDENT_REPO/alpasim_student/" "$ALPASIM_ROOT/plugins/alpasim_student/"

echo "[sync] Copying topology config..."
cp "$STUDENT_REPO/topology/2gpu_alpamayo.yaml" \
   "$ALPASIM_ROOT/src/wizard/configs/topology/2gpu_alpamayo.yaml"

echo "[sync] Copying scripts..."
mkdir -p "$ALPASIM_ROOT/scripts"
cp "$STUDENT_REPO"/scripts/*.py "$ALPASIM_ROOT/scripts/"

echo "[sync] Syncing all workspace packages into venv..."
cd "$ALPASIM_ROOT"
UV_PYTHON_PREFERENCE=system uv sync --extra all --python 3.12

echo "[sync] Done. You can now run:"
echo "  cd $ALPASIM_ROOT"
echo "  uv run --extra all --no-sync python -m alpasim_student.dagger.loop \\"
echo "      --base-dir ./dagger_run --start-iteration 15 --iterations 30 --epochs 50 \\"
echo "      --initial-checkpoint ./dagger_run/checkpoints/student_iter_15_best.pth"
