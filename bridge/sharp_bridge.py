"""
SHARP Bridge Server
===================
Main entry point for the SHARP ↔ TouchDesigner bridge.
Supports two modes:

1. FILE mode: Watches a directory for new frames, runs SHARP, exports PLY files.
   Simplest to set up. TD loads PLY files via Gaussian Splat toolkit.

2. ZMQ mode:  Watches for frames AND publishes Gaussian parameters over ZMQ
   for real-time consumption by a Script DAT in TouchDesigner.

Usage:
    # File-based mode (start here!)
    python sharp_bridge.py --mode file --input-dir ./frames --output-dir ./splats

    # ZMQ + PLY mode (real-time)
    python sharp_bridge.py --mode zmq --port 5555 --input-dir ./frames --output-dir ./splats

    # With custom checkpoint
    python sharp_bridge.py --mode file -c /path/to/sharp_checkpoint.pt
"""

import sys
import time
import struct
import signal
import logging
import argparse
from pathlib import Path

# Add the directory containing this script to sys.path
# This ensures sibling modules (sharp_inference, ply_exporter, etc.) are findable
script_dir = Path(__file__).parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sharp_bridge")


def run_file_mode(args):
    """
    File-based bridge mode.
    Watches input directory for frames → runs SHARP → exports PLY to output directory.
    """
    from sharp_inference import SharpInference
    from ply_exporter import export_gaussians
    from frame_watcher import FrameWatcher

    # Initialize SHARP
    logger.info("Initializing SHARP inference engine...")
    engine = SharpInference(checkpoint_path=args.checkpoint, device=args.device)
    logger.info("SHARP engine ready!")

    # Initialize frame watcher
    watcher = FrameWatcher(args.input_dir, queue_size=2, debounce_ms=args.debounce)
    watcher.start()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        logger.info("Shutdown signal received...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"File mode active. Watching: {args.input_dir} → Output: {args.output_dir}")
    logger.info("Drop images into the input directory or configure TD to export frames there.")

    try:
        while running:
            frame_path = watcher.get_frame(timeout=0.5)
            if frame_path is None:
                continue

            try:
                # Run SHARP inference
                logger.info(f"Processing: {frame_path.name}")
                gaussians = engine.predict_from_file(frame_path)

                # Optionally subsample for performance
                if args.max_points > 0:
                    gaussians = gaussians.subsample(max_points=args.max_points)

                # Export PLY
                ply_name = f"splat_{frame_count:06d}.ply"
                if args.overwrite:
                    ply_name = "current.ply"

                ply_path = export_gaussians(
                    gaussians,
                    output_dir / ply_name,
                    td_coords=True,
                )

                frame_count += 1
                logger.info(
                    f"Frame {frame_count} | {gaussians.n_points} gaussians | "
                    f"{gaussians.inference_time:.3f}s | → {ply_path.name}"
                )

            except Exception as e:
                logger.error(f"Error processing frame: {e}", exc_info=True)

    finally:
        watcher.stop()
        logger.info(f"Bridge stopped. Processed {frame_count} frames.")


def run_zmq_mode(args):
    """
    ZMQ bridge mode.
    Same as file mode, but ALSO publishes Gaussian parameters over ZMQ
    so TD can receive them in real-time via Script DAT.
    """
    import zmq
    from sharp_inference import SharpInference
    from ply_exporter import export_gaussians
    from frame_watcher import FrameWatcher

    # Initialize SHARP
    logger.info("Initializing SHARP inference engine...")
    engine = SharpInference(checkpoint_path=args.checkpoint, device=args.device)
    logger.info("SHARP engine ready!")

    # Initialize ZMQ publisher
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    bind_addr = f"tcp://*:{args.port}"
    socket.bind(bind_addr)
    logger.info(f"ZMQ publisher bound to {bind_addr}")

    # Initialize frame watcher
    watcher = FrameWatcher(args.input_dir, queue_size=2, debounce_ms=args.debounce)
    watcher.start()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        logger.info("Shutdown signal received...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"ZMQ mode active. Watching: {args.input_dir} | ZMQ port: {args.port}")

    try:
        while running:
            frame_path = watcher.get_frame(timeout=0.5)
            if frame_path is None:
                continue

            try:
                # Run SHARP inference
                logger.info(f"Processing: {frame_path.name}")
                gaussians = engine.predict_from_file(frame_path)

                # Subsample if needed
                if args.max_points > 0:
                    gaussians = gaussians.subsample(max_points=args.max_points)

                # Convert to TD coordinates
                td_gaussians = gaussians.to_td_coordinates()

                # --- Publish via ZMQ ---
                n_points = td_gaussians.n_points
                inference_time = td_gaussians.inference_time

                # Pack header: [n_points (uint32), inference_time (float32), frame_id (uint32)]
                header = struct.pack("IfI", n_points, inference_time, frame_count)

                # Pack all parameter arrays as contiguous float32 bytes
                payload = header
                payload += td_gaussians.positions.astype(np.float32).tobytes()
                payload += td_gaussians.colors.astype(np.float32).tobytes()
                payload += td_gaussians.scales.astype(np.float32).tobytes()
                payload += td_gaussians.rotations.astype(np.float32).tobytes()
                payload += td_gaussians.opacities.astype(np.float32).tobytes()

                socket.send(payload)

                # --- Also export PLY (optional, for Gaussian Splat toolkit) ---
                if args.output_dir:
                    ply_path = export_gaussians(
                        gaussians,
                        output_dir / "current.ply",
                        td_coords=True,
                    )

                frame_count += 1
                logger.info(
                    f"Frame {frame_count} | {n_points} gaussians | "
                    f"{inference_time:.3f}s | ZMQ sent"
                )

            except Exception as e:
                logger.error(f"Error processing frame: {e}", exc_info=True)

    finally:
        watcher.stop()
        socket.close()
        context.term()
        logger.info(f"Bridge stopped. Processed {frame_count} frames.")


def main():
    parser = argparse.ArgumentParser(
        description="SHARP ↔ TouchDesigner Bridge Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # File-based mode (simplest — start here!)
  python sharp_bridge.py --mode file --input-dir ./frames --output-dir ./splats

  # ZMQ real-time mode
  python sharp_bridge.py --mode zmq --port 5555

  # With custom checkpoint and GPU selection
  python sharp_bridge.py --mode file -c sharp_2572gikvuh.pt --device cuda:1
        """,
    )

    parser.add_argument(
        "--mode", choices=["file", "zmq"], default="file",
        help="Bridge mode: 'file' (PLY export only) or 'zmq' (real-time + PLY)."
    )
    parser.add_argument(
        "--input-dir", default="./frames",
        help="Directory to watch for input frames from TouchDesigner."
    )
    parser.add_argument(
        "--output-dir", default="./splats",
        help="Directory to export Gaussian PLY files."
    )
    parser.add_argument(
        "-c", "--checkpoint", default=None,
        help="Path to SHARP checkpoint file. Auto-downloads if not specified."
    )
    parser.add_argument(
        "--device", default="cuda",
        help="PyTorch device for inference: 'cuda', 'cuda:0', 'cpu', 'mps'."
    )
    parser.add_argument(
        "-p", "--port", type=int, default=5555,
        help="ZMQ publisher port (only used in zmq mode)."
    )
    parser.add_argument(
        "--max-points", type=int, default=0,
        help="Max Gaussians to output. 0 = all points (can be 100k+)."
    )
    parser.add_argument(
        "--debounce", type=int, default=100,
        help="Frame debounce in milliseconds. Prevents processing duplicate events."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="In file mode, overwrite 'current.ply' instead of sequential numbering."
    )

    args = parser.parse_args()

    if args.mode == "file":
        run_file_mode(args)
    elif args.mode == "zmq":
        run_zmq_mode(args)


if __name__ == "__main__":
    main()
