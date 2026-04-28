"""
TouchDesigner Script DAT: ZMQ Gaussian Receiver
================================================
Receives Gaussian Splat parameters from the SHARP Bridge Server over ZMQ.
Store this script in a Script CHOP or Script DAT in your TD project.

SETUP IN TOUCHDESIGNER:
1. Create a Script CHOP or Timer CHOP
2. Paste this code into the Script DAT associated with it
3. Configure the ZMQ address if needed (default: tcp://localhost:5555)
4. Create a 'gaussian_store' Base COMP to store the received data

NOTE: pyzmq must be installed in TouchDesigner's Python.
You can install it from TD's Python shell:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyzmq', 'numpy'])
"""

import struct
import numpy as np

# ============================================================
# CONFIGURATION — edit these to match your setup
# ============================================================
ZMQ_ADDRESS = "tcp://localhost:5555"
STORE_OP_PATH = "gaussian_store"  # Base COMP to store data in
# ============================================================

# Global ZMQ socket (initialized once)
_zmq_socket = None
_zmq_context = None


def _init_zmq():
    """Initialize ZMQ subscriber socket (called once)."""
    global _zmq_socket, _zmq_context

    if _zmq_socket is not None:
        return

    try:
        import zmq
    except ImportError:
        print("[ZMQ Receiver] ERROR: pyzmq not installed!")
        print("  Run in TD Python shell: ")
        print("    import subprocess, sys")
        print("    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyzmq'])")
        return

    _zmq_context = zmq.Context()
    _zmq_socket = _zmq_context.socket(zmq.SUB)
    _zmq_socket.connect(ZMQ_ADDRESS)
    _zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    _zmq_socket.setsockopt(zmq.RCVTIMEO, 10)  # 10ms timeout — non-blocking for TD
    _zmq_socket.setsockopt(zmq.CONFLATE, 1)    # Only keep the latest message

    print(f"[ZMQ Receiver] Connected to {ZMQ_ADDRESS}")


def onCook(scriptOp):
    """
    Called every frame by TouchDesigner.
    Receives the latest Gaussian parameters from the SHARP bridge.
    """
    global _zmq_socket

    _init_zmq()
    if _zmq_socket is None:
        return

    import zmq

    try:
        data = _zmq_socket.recv(flags=zmq.NOBLOCK)
    except zmq.Again:
        return  # No new data this frame — keep previous Gaussians

    # ---- Parse Header ----
    # Header: [n_points (uint32), inference_time (float32), frame_id (uint32)]
    header_fmt = "IfI"
    header_size = struct.calcsize(header_fmt)

    if len(data) < header_size:
        return

    n_points, inference_time, frame_id = struct.unpack(header_fmt, data[:header_size])
    offset = header_size

    # ---- Parse Gaussian Arrays ----
    # All arrays are packed as contiguous float32

    # Positions: (N, 3)
    pos_bytes = n_points * 3 * 4
    positions = np.frombuffer(data[offset:offset + pos_bytes], dtype=np.float32).reshape(n_points, 3)
    offset += pos_bytes

    # Colors: (N, 3)
    col_bytes = n_points * 3 * 4
    colors = np.frombuffer(data[offset:offset + col_bytes], dtype=np.float32).reshape(n_points, 3)
    offset += col_bytes

    # Scales: (N, 3)
    scl_bytes = n_points * 3 * 4
    scales = np.frombuffer(data[offset:offset + scl_bytes], dtype=np.float32).reshape(n_points, 3)
    offset += scl_bytes

    # Rotations: (N, 4)
    rot_bytes = n_points * 4 * 4
    rotations = np.frombuffer(data[offset:offset + rot_bytes], dtype=np.float32).reshape(n_points, 4)
    offset += rot_bytes

    # Opacities: (N, 1)
    opa_bytes = n_points * 1 * 4
    opacities = np.frombuffer(data[offset:offset + opa_bytes], dtype=np.float32).reshape(n_points, 1)

    # ---- Store in TD ----
    store = op(STORE_OP_PATH)
    if store is None:
        print(f"[ZMQ Receiver] WARNING: '{STORE_OP_PATH}' op not found. Create a Base COMP with this name.")
        return

    store.store("positions", positions)
    store.store("colors", colors)
    store.store("scales", scales)
    store.store("rotations", rotations)
    store.store("opacities", opacities)
    store.store("n_points", n_points)
    store.store("inference_time", inference_time)
    store.store("frame_id", frame_id)

    # Update a status text DAT if it exists
    status_op = op("sharp_status")
    if status_op:
        status_op.clear()
        status_op.appendRow(["Frame", str(frame_id)])
        status_op.appendRow(["Points", str(n_points)])
        status_op.appendRow(["Inference (s)", f"{inference_time:.3f}"])


def onSetupParameters(scriptOp):
    """Called when the script first loads."""
    _init_zmq()


def onPulse(par):
    """Handle pulse parameters."""
    pass
