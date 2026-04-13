#!/usr/bin/env bash
# sync.sh — Pull latest code from GitHub and copy into the alpasim repo.
# Run this on the server after pushing changes from your Windows machine.
#
# Usage:
#   cd /home/harun/alpasim-student
#   ./sync.sh

set -e

STUDENT_REPO="/home/harun/alpasim-student"
ALPASIM_ROOT="/home/harun/alpasim"

echo "[sync] Pulling latest from GitHub..."
cd "$STUDENT_REPO"
git pull

echo "[sync] Copying plugin into alpasim..."
cp -r "$STUDENT_REPO/alpasim_student" "$ALPASIM_ROOT/plugins/"

echo "[sync] Copying topology config..."
cp "$STUDENT_REPO/topology/2gpu_alpamayo.yaml" \
   "$ALPASIM_ROOT/src/wizard/configs/topology/2gpu_alpamayo.yaml"

echo "[sync] Done. You can now run:"
echo "  cd $ALPASIM_ROOT"
echo "  uv run --extra all --no-sync python -m alpasim_student.dagger.loop \\"
echo "      --base-dir ./dagger_run_v3 --iterations 7 --epochs 30"
