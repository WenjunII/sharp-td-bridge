"""
TouchDesigner Script SOP: Gaussian Point Cloud Generator
========================================================
Reads Gaussian parameters from the gaussian_store Base COMP
and generates a point cloud SOP for rendering via instancing.

SETUP IN TOUCHDESIGNER:
1. Create a Script SOP
2. Paste this code into its Script DAT
3. Ensure 'gaussian_store' Base COMP exists and is populated by the ZMQ receiver
4. Connect the Script SOP output to an Instancing Geometry COMP or Particle SOP

The Script SOP creates points with:
- Position (P): from Gaussian positions
- Color (Cd): from Gaussian colors
- Scale (custom attrib 'pscale'): from Gaussian scales
- Normal (N): derived from quaternion rotation
- Alpha (Alpha): from Gaussian opacities
"""

import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
STORE_OP_PATH = "gaussian_store"
MAX_DISPLAY_POINTS = 50000  # Cap for TD performance
# ============================================================


def onCook(scriptOp):
    """
    Called every frame. Generates point cloud geometry from stored Gaussian data.
    """
    scriptOp.clear()

    # Fetch data from store
    store = op(STORE_OP_PATH)
    if store is None:
        return

    positions = store.fetch("positions")
    colors = store.fetch("colors")
    scales = store.fetch("scales")
    opacities = store.fetch("opacities")
    rotations = store.fetch("rotations")

    if positions is None or len(positions) == 0:
        return

    n = len(positions)

    # Subsample if too many points for smooth TD performance
    if n > MAX_DISPLAY_POINTS:
        # Keep the most opaque points
        if opacities is not None:
            indices = np.argsort(opacities[:, 0])[-MAX_DISPLAY_POINTS:]
        else:
            indices = np.random.choice(n, MAX_DISPLAY_POINTS, replace=False)
        positions = positions[indices]
        colors = colors[indices] if colors is not None else None
        scales = scales[indices] if scales is not None else None
        opacities = opacities[indices] if opacities is not None else None
        rotations = rotations[indices] if rotations is not None else None
        n = MAX_DISPLAY_POINTS

    # Create points
    for i in range(n):
        pt = scriptOp.appendPoint()
        pt.x = float(positions[i, 0])
        pt.y = float(positions[i, 1])
        pt.z = float(positions[i, 2])

    # Add color attribute
    if colors is not None:
        cd_attrib = scriptOp.pointAttribs.create("Cd", (1.0, 1.0, 1.0))
        for i, pt in enumerate(scriptOp.points):
            pt.Cd = (float(colors[i, 0]), float(colors[i, 1]), float(colors[i, 2]))

    # Add per-point scale attribute (average of 3-axis scales)
    if scales is not None:
        pscale_attrib = scriptOp.pointAttribs.create("pscale", 0.01)
        for i, pt in enumerate(scriptOp.points):
            avg_scale = float(np.mean(np.abs(scales[i])))
            pt.pscale = avg_scale

    # Add alpha/opacity attribute
    if opacities is not None:
        alpha_attrib = scriptOp.pointAttribs.create("Alpha", 1.0)
        for i, pt in enumerate(scriptOp.points):
            pt.Alpha = float(opacities[i, 0])

    # Add per-axis scale for instancing (Geometry COMP can read these)
    if scales is not None:
        sx_attrib = scriptOp.pointAttribs.create("sx", 0.01)
        sy_attrib = scriptOp.pointAttribs.create("sy", 0.01)
        sz_attrib = scriptOp.pointAttribs.create("sz", 0.01)
        for i, pt in enumerate(scriptOp.points):
            pt.sx = float(np.exp(scales[i, 0]))  # SHARP stores log-scale
            pt.sy = float(np.exp(scales[i, 1]))
            pt.sz = float(np.exp(scales[i, 2]))

    # Add quaternion rotation for instancing
    if rotations is not None:
        rw_attrib = scriptOp.pointAttribs.create("rw", 1.0)
        rx_attrib = scriptOp.pointAttribs.create("rx", 0.0)
        ry_attrib = scriptOp.pointAttribs.create("ry", 0.0)
        rz_attrib = scriptOp.pointAttribs.create("rz", 0.0)
        for i, pt in enumerate(scriptOp.points):
            pt.rw = float(rotations[i, 0])
            pt.rx = float(rotations[i, 1])
            pt.ry = float(rotations[i, 2])
            pt.rz = float(rotations[i, 3])


def onSetupParameters(scriptOp):
    """Called when the script loads."""
    print(f"[Point Cloud Gen] Configured. Store: {STORE_OP_PATH}, Max points: {MAX_DISPLAY_POINTS}")
