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
echo "      --base-dir ./dagger_run --start-iteration 15 --iterations 30 --epochs 50 \\"
echo "      --initial-checkpoint ./dagger_run/checkpoints/student_iter_15_best.pth \\"
echo "      --scenes clipgt-01d503d4-449b-46fc-8d78-9085e70d3554 \\"
echo "               clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb \\"
echo "               clipgt-804afc4a-fd1e-4f58-bd39-a4c486a916e5"
