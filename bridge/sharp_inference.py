"""
SHARP Inference Wrapper
=======================
Wraps Apple's SHARP model for programmatic inference.
Loads the model once, keeps it warm on GPU, and provides a simple predict() API.

This module uses SHARP's internal API (from the cloned apple/ml-sharp repo).
The SHARP repo must be installed in the conda environment: pip install -r requirements.txt
from within the ml-sharp directory.
"""

import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class GaussianParams:
    """Container for 3D Gaussian Splat parameters output by SHARP."""

    positions: np.ndarray    # (N, 3) float32 — XYZ in OpenCV convention
    colors: np.ndarray       # (N, 3) float32 — RGB [0,1]
    scales: np.ndarray       # (N, 3) float32 — scale per axis (log-space)
    rotations: np.ndarray    # (N, 4) float32 — quaternion (w, x, y, z)
    opacities: np.ndarray    # (N, 1) float32 — sigmoid-activated alpha
    n_points: int = field(init=False)
    inference_time: float = 0.0

    def __post_init__(self):
        self.n_points = self.positions.shape[0]

    def to_td_coordinates(self) -> "GaussianParams":
        """
        Convert from SHARP's OpenCV convention (x right, y down, z forward)
        to TouchDesigner convention (x right, y up, z backward).

        Returns a new GaussianParams with transformed positions.
        """
        td_positions = self.positions.copy()
        td_positions[:, 1] *= -1  # Flip Y (down → up)
        td_positions[:, 2] *= -1  # Flip Z (forward → backward)

        # Re-center around origin (SHARP places scene at ~(0, 0, +z))
        centroid = td_positions.mean(axis=0)
        td_positions -= centroid

        return GaussianParams(
            positions=td_positions,
            colors=self.colors,
            scales=self.scales,
            rotations=self.rotations,
            opacities=self.opacities,
            inference_time=self.inference_time,
        )

    def subsample(self, max_points: int = 50000, by_opacity: bool = True) -> "GaussianParams":
        """
        Subsample Gaussians to reduce load for TD rendering.
        """
        if self.n_points <= max_points:
            return self

        if by_opacity:
            # Flatten opacities for sorting
            flat_opacities = self.opacities.flatten()
            indices = np.argsort(flat_opacities)[-max_points:]
        else:
            indices = np.random.choice(self.n_points, max_points, replace=False)

        return GaussianParams(
            positions=self.positions[indices],
            colors=self.colors[indices],
            scales=self.scales[indices],
            rotations=self.rotations[indices],
            opacities=self.opacities[indices] if self.opacities.ndim > 1 else self.opacities[indices, np.newaxis],
            inference_time=self.inference_time,
        )


class SharpInference:
    """
    Wrapper around Apple's SHARP model for single-image → Gaussian Splat inference.

    Usage:
        engine = SharpInference(device="cuda")
        gaussians = engine.predict(image_array)
    """

    def __init__(self, checkpoint_path: str | None = None, device: str = "cuda"):
        """
        Initialize the SHARP inference engine.

        Args:
            checkpoint_path: Path to the SHARP checkpoint file (.pt).
                           If None, the model will auto-download on first use.
            device: Device to run inference on ("cuda", "cpu", "mps").
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SHARP predictor model."""
        try:
            # Import SHARP's predictor — this is the main model class
            # that handles image → Gaussian parameter regression.
            from sharp.models import PredictorParams, create_predictor

            logger.info("Loading SHARP model...")

            # Get the default model configuration
            params = PredictorParams()

            # Build the predictor model
            self.model = create_predictor(params)

            # Load checkpoint weights
            if self.checkpoint_path:
                ckpt_path = Path(self.checkpoint_path)
                logger.info(f"Loading checkpoint from {ckpt_path}")
                state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            else:
                # SHARP auto-downloads via torch.hub
                DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
                logger.info(f"No checkpoint provided. Downloading default model from {DEFAULT_MODEL_URL}")
                state_dict = torch.hub.load_state_dict_from_url(
                    DEFAULT_MODEL_URL, 
                    map_location=self.device,
                    progress=True
                )

            # Handle nested state dicts (some checkpoints wrap in 'model' key)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            self.model.load_state_dict(state_dict, strict=True)
            self.model = self.model.to(self.device).eval()

            logger.info(f"SHARP model loaded successfully on {self.device}")

        except ImportError as e:
            logger.error(
                f"Failed to import SHARP modules: {e}\n"
                "Make sure SHARP is installed: cd ml-sharp && pip install -r requirements.txt"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load SHARP model: {e}")
            raise

    def _get_focal_length(self, width: float, height: float, f_mm: float = 30.0) -> float:
        """Convert focal length from mm (35mm equivalent) to pixels."""
        return f_mm * np.sqrt(width**2.0 + height**2.0) / np.sqrt(36**2 + 24**2)

    def predict(self, image: np.ndarray | Image.Image) -> GaussianParams:
        """
        Run SHARP inference on a single image.

        Args:
            image: Input image as HxWx3 uint8 numpy array (RGB) or PIL Image.

        Returns:
            GaussianParams with positions, colors, scales, rotations, opacities.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        import torch.nn.functional as F
        
        # Match SHARP's internal resolution for best quality
        internal_shape = (1536, 1536)
        
        # Calculate disparity_factor (focal length relative to width)
        f_px = self._get_focal_length(pil_img.width, pil_img.height)
        disparity_factor = torch.tensor([f_px / pil_img.width]).float().to(self.device)

        t0 = time.time()

        with torch.no_grad():
            # Preprocess image to tensor (permute and normalize to [0, 1])
            img_pt = torch.from_numpy(np.array(pil_img.convert("RGB"))).float().to(self.device).permute(2, 0, 1) / 255.0
            
            # Resize to model's expected internal resolution
            img_tensor = F.interpolate(
                img_pt[None],
                size=(internal_shape[1], internal_shape[0]),
                mode="bilinear",
                align_corners=True,
            )

            # Forward pass through the model with required disparity_factor
            output = self.model(img_tensor, disparity_factor)

            # Extract Gaussian parameters from model output (Gaussians3D NamedTuple)
            # 0: mean_vectors, 1: singular_values, 2: quaternions, 3: colors, 4: opacities
            positions = self._extract_param(output, ["mean_vectors", "means", "positions", "xyz"], idx=0)
            scales = self._extract_param(output, ["singular_values", "scales", "log_scales"], idx=1)
            rotations = self._extract_param(output, ["quaternions", "rotations", "quats"], idx=2)
            colors = self._extract_param(output, ["colors", "sh", "rgb"], idx=3)
            opacities = self._extract_param(output, ["opacities", "opacity", "alpha"], idx=4)

        inference_time = time.time() - t0

        # Convert to numpy
        positions_np = positions[0].cpu().numpy().astype(np.float32)
        colors_np = colors[0].cpu().numpy().astype(np.float32)
        scales_np = scales[0].cpu().numpy().astype(np.float32)
        rotations_np = rotations[0].cpu().numpy().astype(np.float32)
        opacities_np = opacities[0].cpu().numpy().astype(np.float32)

        # Reshape positions and other params if they are [H, W, C]
        if positions_np.ndim == 3: # (H, W, 3)
            positions_np = positions_np.reshape(-1, 3)
            colors_np = colors_np.reshape(-1, 3)
            scales_np = scales_np.reshape(-1, 3)
            rotations_np = rotations_np.reshape(-1, 4)
            opacities_np = opacities_np.reshape(-1, 1)

        return GaussianParams(
            positions=positions_np,
            colors=colors_np,
            scales=scales_np,
            rotations=rotations_np,
            opacities=opacities_np,
            inference_time=inference_time,
        )

    def _extract_param(self, output, possible_keys: list[str], idx: int | None = None) -> torch.Tensor:
        """Extract a parameter from the model output, trying attribute names then index."""
        # Try attribute names first (NamedTuple/SimpleNamespace)
        for key in possible_keys:
            if hasattr(output, key):
                return getattr(output, key)
        
        # Try dictionary keys
        if isinstance(output, dict):
            for key in possible_keys:
                if key in output:
                    return output[key]
        
        # Fallback to index if provided
        if idx is not None and isinstance(output, (tuple, list)):
            if idx < len(output):
                return output[idx]
            
        raise ValueError(f"Could not extract parameter {possible_keys} (idx {idx}) from {type(output)}")

    def predict_from_file(self, image_path: str | Path) -> GaussianParams:
        """
        Convenience: predict from an image file path.
        Includes a retry loop to handle Windows file locks during save/copy.
        """
        max_retries = 10
        retry_delay = 0.1
        
        for i in range(max_retries):
            try:
                img = Image.open(image_path).convert("RGB")
                return self.predict(img)
            except (PermissionError, OSError) as e:
                if i < max_retries - 1:
                    logger.debug(f"File locked, retrying in {retry_delay}s... ({i+1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to open image after {max_retries} retries: {e}")
                    raise
        
        # Fallback (should not be reached)
        raise RuntimeError("Failed to process file due to lock.")
