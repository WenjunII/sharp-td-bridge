"""
TouchDesigner Script DAT: Splat PLY Loader
===========================================
Watches the splats directory for updated PLY files
and triggers a reload on the Gaussian Splat rendering component.

SETUP IN TOUCHDESIGNER:
1. Create a Timer CHOP set to fire every ~0.5 seconds
2. Connect it to a CHOP Execute DAT
3. In the CHOP Execute, call this script's check_for_update() function
4. Set SPLAT_COMP_PATH to point to your Gaussian Splat toolkit component

Alternative: Use this as a standalone Script DAT with a Timer callback.
"""

import os
import time

# ============================================================
# CONFIGURATION
# ============================================================
SPLAT_DIR = "C:/sharp-td-bridge/splats"         # Directory where bridge outputs PLY files
SPLAT_FILENAME = "current.ply"                   # PLY file to watch
SPLAT_COMP_PATH = "gaussian_splat"               # Path to GaussianSplatGEO or TDGS comp in TD
RELOAD_PARAM_NAME = "Reload"                     # Name of the reload pulse parameter
# ============================================================

_last_mod_time = 0


def check_for_update():
    """
    Check if the PLY file has been updated since last check.
    If so, trigger a reload on the Gaussian Splat component.
    """
    global _last_mod_time

    ply_path = os.path.join(SPLAT_DIR, SPLAT_FILENAME)

    if not os.path.exists(ply_path):
        return False

    try:
        current_mod_time = os.path.getmtime(ply_path)
    except OSError:
        return False  # File might be mid-write

    if current_mod_time <= _last_mod_time:
        return False  # No change

    _last_mod_time = current_mod_time

    # Trigger reload on the Gaussian Splat component
    splat_comp = op(SPLAT_COMP_PATH)
    if splat_comp is None:
        print(f"[Splat Loader] WARNING: '{SPLAT_COMP_PATH}' not found in project")
        return False

    # Try to pulse the reload parameter
    try:
        if hasattr(splat_comp.par, RELOAD_PARAM_NAME):
            getattr(splat_comp.par, RELOAD_PARAM_NAME).pulse()
            print(f"[Splat Loader] Reloaded {SPLAT_FILENAME} (mod: {current_mod_time:.1f})")
            return True
        else:
            # Try common alternative parameter names
            for param_name in ["Reload", "reload", "Reloadfile", "Loadfile", "reload1"]:
                if hasattr(splat_comp.par, param_name):
                    getattr(splat_comp.par, param_name).pulse()
                    print(f"[Splat Loader] Reloaded via '{param_name}'")
                    return True

            # If no reload param found, try updating the file path parameter
            for param_name in ["File", "file", "Plyfile", "plyfile", "Filepath"]:
                if hasattr(splat_comp.par, param_name):
                    getattr(splat_comp.par, param_name).val = ply_path
                    print(f"[Splat Loader] Updated file path param '{param_name}'")
                    return True

            print(f"[Splat Loader] WARNING: Could not find reload parameter on '{SPLAT_COMP_PATH}'")
            return False

    except Exception as e:
        print(f"[Splat Loader] Error triggering reload: {e}")
        return False


# ---- Timer/CHOP Execute callbacks ----

def onOffToOn(channel, sampleIndex, val, prev):
    """Called when Timer CHOP triggers. Check for PLY updates."""
    check_for_update()


def onValueChange(channel, sampleIndex, val, prev):
    """Alternative callback for value-based triggers."""
    pass
