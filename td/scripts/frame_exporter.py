"""
TouchDesigner Script TOP: Frame Exporter
=========================================
Exports the current StreamDiffusionTD output frame to disk
so the SHARP Bridge Server can pick it up.

SETUP IN TOUCHDESIGNER:
1. Create a Script TOP
2. Set its input to the StreamDiffusionTD output TOP
3. Paste this code into the Script DAT
4. Set the export path to match the bridge's --input-dir

The script writes the frame as PNG to a specified directory.
It uses a throttle to avoid exporting faster than SHARP can process.
"""

import os
import time

# ============================================================
# CONFIGURATION
# ============================================================
EXPORT_DIR = "C:/sharp-td-bridge/frames"       # Must match bridge --input-dir
EXPORT_FILENAME = "current_frame.png"           # Filename the bridge watches for
MIN_INTERVAL_SEC = 0.5                          # Min seconds between exports
# ============================================================

_last_export_time = 0


def onCook(scriptOp):
    """
    Called every frame. Exports the input TOP to disk at a throttled rate.
    """
    global _last_export_time

    # Throttle: don't export faster than SHARP can process
    now = time.time()
    if now - _last_export_time < MIN_INTERVAL_SEC:
        return

    # Get input TOP
    input_top = scriptOp.inputs[0] if scriptOp.inputs else None
    if input_top is None:
        return

    # Ensure export directory exists
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Export the frame
    export_path = os.path.join(EXPORT_DIR, EXPORT_FILENAME)

    try:
        # Use TD's built-in save() method on the TOP
        # This writes the current frame to disk as PNG
        input_top.save(export_path)
        _last_export_time = now
    except Exception as e:
        print(f"[Frame Exporter] Error saving frame: {e}")


def onSetupParameters(scriptOp):
    """Called when the script loads."""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    print(f"[Frame Exporter] Exporting to: {EXPORT_DIR}/{EXPORT_FILENAME}")
    print(f"[Frame Exporter] Interval: {MIN_INTERVAL_SEC}s")
