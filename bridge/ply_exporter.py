"""
PLY Exporter for Gaussian Splats
=================================
Exports SHARP Gaussian parameters to PLY format compatible with:
- TouchDesigner Gaussian Splat toolkits (Tim Gerritsen / TDGS)
- Standard 3DGS renderers
- SuperSplat web viewer

Uses double-buffering to prevent file read/write conflicts
when TouchDesigner is continuously loading updated PLY files.
"""

import os
import shutil
import logging
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

logger = logging.getLogger(__name__)


# Standard 3DGS PLY vertex properties
GAUSSIAN_PLY_DTYPE = [
    ("x", "f4"), ("y", "f4"), ("z", "f4"),           # Position
    ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),         # Normal (unused, set to 0)
    ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),  # DC color (RGB)
    ("opacity", "f4"),                                  # Opacity (pre-sigmoid)
    ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),  # Log-scale
    ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),  # Quaternion
]


def gaussians_to_ply_array(positions, colors, scales, rotations, opacities):
    """
    Convert raw Gaussian parameter arrays to a structured numpy array
    matching the standard 3DGS PLY format.

    Args:
        positions:  (N, 3) float32 — XYZ
        colors:     (N, 3) float32 — RGB in [0, 1]
        scales:     (N, 3) float32 — per-axis scale (log-space)
        rotations:  (N, 4) float32 — quaternion (w, x, y, z)
        opacities:  (N, 1) float32 — opacity [0, 1]

    Returns:
        Structured numpy array with 3DGS PLY fields.
    """
    n = positions.shape[0]
    arr = np.zeros(n, dtype=GAUSSIAN_PLY_DTYPE)

    # Positions
    arr["x"] = positions[:, 0]
    arr["y"] = positions[:, 1]
    arr["z"] = positions[:, 2]

    # Normals (unused, set to zero)
    arr["nx"] = 0.0
    arr["ny"] = 0.0
    arr["nz"] = 0.0

    # DC color coefficients (spherical harmonics band 0)
    # Standard 3DGS stores these as SH coefficients, not raw RGB.
    # The DC coefficient = (color - 0.5) / C0, where C0 = 0.28209479177...
    # But many viewers expect raw RGB or a simpler mapping.
    # We'll store the direct values and let the viewer handle the conversion.
    arr["f_dc_0"] = colors[:, 0]
    arr["f_dc_1"] = colors[:, 1]
    arr["f_dc_2"] = colors[:, 2]

    # Opacity (pre-sigmoid inverse for standard 3DGS format)
    # Some viewers expect raw sigmoid-activated values, others expect logits.
    # We store the raw values from SHARP.
    opacities_flat = opacities.flatten() if opacities.ndim > 1 else opacities
    arr["opacity"] = opacities_flat

    # Log-scale
    arr["scale_0"] = scales[:, 0]
    arr["scale_1"] = scales[:, 1]
    arr["scale_2"] = scales[:, 2]

    # Quaternion rotation (w, x, y, z)
    arr["rot_0"] = rotations[:, 0]
    arr["rot_1"] = rotations[:, 1]
    arr["rot_2"] = rotations[:, 2]
    arr["rot_3"] = rotations[:, 3]

    return arr


def export_ply(
    positions: np.ndarray,
    colors: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    opacities: np.ndarray,
    output_path: str | Path,
    double_buffer: bool = True,
) -> Path:
    """
    Export Gaussian parameters to a PLY file.

    Args:
        positions:  (N, 3) float32
        colors:     (N, 3) float32
        scales:     (N, 3) float32
        rotations:  (N, 4) float32
        opacities:  (N, 1) float32
        output_path: Destination PLY file path.
        double_buffer: If True, write to a temp file first then atomically rename.
                      Prevents TouchDesigner from reading a partially-written file.

    Returns:
        Path to the written PLY file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build structured array
    arr = gaussians_to_ply_array(positions, colors, scales, rotations, opacities)

    # Create PLY element
    vertex_element = PlyElement.describe(arr, "vertex")
    ply_data = PlyData([vertex_element])

    if double_buffer:
        # Write to temp file first, then atomic rename
        temp_path = output_path.with_suffix(".ply.tmp")
        ply_data.write(str(temp_path))

        # Atomic replace (on Windows, need to remove dest first)
        if output_path.exists():
            output_path.unlink()
        temp_path.rename(output_path)
    else:
        ply_data.write(str(output_path))

    logger.debug(f"Exported {len(arr)} Gaussians to {output_path}")
    return output_path


def export_gaussians(gaussian_params, output_path: str | Path, td_coords: bool = True) -> Path:
    """
    Convenience wrapper: export a GaussianParams object to PLY.

    Args:
        gaussian_params: GaussianParams dataclass from sharp_inference.py
        output_path: Destination PLY file path.
        td_coords: If True, convert to TouchDesigner coordinate system first.

    Returns:
        Path to the written PLY file.
    """
    if td_coords:
        gaussian_params = gaussian_params.to_td_coordinates()

    return export_ply(
        positions=gaussian_params.positions,
        colors=gaussian_params.colors,
        scales=gaussian_params.scales,
        rotations=gaussian_params.rotations,
        opacities=gaussian_params.opacities,
        output_path=output_path,
    )
