# SHARP ↔ TouchDesigner Bridge

Real-time 3D Gaussian Splatting pipeline for TouchDesigner. This bridge connects **StreamDiffusionTD** (or any image source in TD) to **Apple's SHARP** model, generating high-quality 3D particles from single 2D frames in real-time.

## Features

- **Real-time 3D Reconstruction**: Convert 2D images to 3D Gaussian Splats in ~2.5s.
- **GPU Accelerated**: Optimized for NVIDIA RTX GPUs using CUDA.
- **Auto-Subsampling**: Smartly downsamples to a target point count (e.g., 25k-100k) for smooth rendering.
- **Automatic Camera Estimation**: Calculates focal length and disparity factors internally.
- **TouchDesigner Integration**: Ready-to-use scripts for frame exporting and splat loading.
- **Windows Optimized**: Handles file locks and path issues common in TD environments.

## Installation

### 1. Prerequisites
- **Python**: Anaconda or Miniconda installed.
- **GPU**: NVIDIA RTX GPU (RTX 3080 Ti Laptop tested).
- **Git**: Installed on your system.

### 2. One-Click Setup
Run the setup script in PowerShell to create the environment and clone dependencies:
```powershell
./setup_env.ps1
```

### 3. GPU (CUDA) Support Fix
If the setup installs the CPU version of PyTorch by default, run this to enable GPU support:
```powershell
conda run -n sharp pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### 4. Download SHARP Model
Run this to download the 2.6GB model checkpoint:
```powershell
New-Item -ItemType Directory -Force -Path "$HOME\.cache\torch\hub\checkpoints"
curl.exe -L "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt" -o "$HOME\.cache\torch\hub\checkpoints\sharp_2572gikvuh.pt"
```

## Usage

### Start the Bridge
Always use the **absolute path** to your environment's Python to avoid path conflicts. 

**Recommended settings for Laptop GPUs (RTX 3080 Ti):**
```powershell
# Use absolute path to your conda environment's python.exe
C:\Users\wenju\anaconda3\envs\sharp\python.exe -u bridge/sharp_bridge.py --mode file --max-points 25000 --overwrite
```
- `--max-points 25000`: Essential for keeping TouchDesigner FPS high.
- `--overwrite`: Always updates `current.ply` for easier TD loading.
- `-u`: Enables unbuffered output so you can see logs instantly.

## TouchDesigner Integration

### 1. Export Frames
Use `td/scripts/frame_exporter.py` in a Script TOP. 
**Note**: Always use forward slashes `/` in your paths to avoid `unicodeescape` errors.
```python
# Use forward slashes!
EXPORT_DIR = "C:/Users/wenju/.gemini/antigravity/scratch/sharp-td-bridge/frames"
```

### 2. Rendering Splats
Use a Gaussian Splat .tox and point it to `splats/current.ply`. 
**Performance Tip**: Turn **OFF** "Auto Sort" or "Depth Sort" in your TD component if your FPS is low.

## Troubleshooting

- **Unicode Error**: Use `/` instead of `\` in Python strings within TouchDesigner.
- **Permission Denied**: The bridge now includes a retry loop, but ensure no other app has the `.ply` file locked for long periods.
- **Low FPS**: Lower the `--max-points` argument and decrease the frequency of your Timer CHOP reload.

## License
[MIT](LICENSE) - Created by Wenjun (2026)

## Acknowledgements
- [Apple ML-SHARP](https://github.com/apple/ml-sharp)
- [StreamDiffusionTD](https://github.com/dotsimulate/StreamDiffusionTD)
