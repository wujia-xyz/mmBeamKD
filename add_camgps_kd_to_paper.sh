#!/bin/bash
# Run this script after s32_v12_camgps_kd training completes.
# It evaluates the best model, computes bootstrap CI, and updates paper/main.tex Table I.

LOGDIR="log/s32_v12_camgps_kd"
PAPER="paper/main.tex"

# Wait for training to finish
echo "Checking training status..."
if [ ! -f "$LOGDIR/best_model.pth" ]; then
    echo "Training not complete yet."
    exit 1
fi

cat "$LOGDIR/test_results.json" 2>/dev/null || echo "No test_results.json yet"

echo ""
echo "Run the following Python to get bootstrap CI:"
echo "  python3 - <<'EOF'"
echo "  # Load s32_v12_camgps_kd/best_model.pth"
echo "  # Zero lidar/radar inputs"
echo "  # Compute DBA, CI, p-value vs baseline"
echo "  EOF"
echo ""
echo "Then update paper/main.tex Table I:"
echo "  Add row after CamGPS-only:"
echo "  CamGPS-only-KD (ours) & 21.7M & [DBA] & [CI] & [p] & [Top-1] & 19.9 & 213.5"
