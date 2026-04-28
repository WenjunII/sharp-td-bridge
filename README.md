# SHARP ↔ TouchDesigner Bridge

Real-time 3D Gaussian Splatting pipeline for TouchDesigner. This bridge connects **StreamDiffusionTD** (or any image source in TD) to **Apple's SHARP** model, generating high-quality 3D particles from single 2D frames in real-time.

## Features

- **Real-time 3D Reconstruction**: Convert 2D images to 3D Gaussian Splats in ~2.5s.
- **GPU Accelerated**: Optimized for NVIDIA RTX GPUs using CUDA.
- **Auto-Subsampling**: Smartly downsamples to a target point count (e.g., 100k) for smooth rendering in TouchDesigner.
- **Automatic Camera Estimation**: Calculates focal length and disparity factors internally.
- **TouchDesigner Integration**: Ready-to-use scripts for frame exporting and splat loading.
- **Double-Buffered Export**: Prevents file read/write conflicts on Windows.

## Installation

### 1. Prerequisites
- **Python**: Anaconda or Miniconda installed.
- **GPU**: NVIDIA RTX GPU with CUDA 11.8+ (CUDA 12.4 recommended).
- **Git**: Installed on your system.

### 2. One-Click Setup
Run the setup script in PowerShell to create the environment and clone dependencies:
```powershell
./setup_env.ps1
```

### 3. Manual Fix for GPU Support
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
Use the absolute path to your environment's Python to ensure correct dependency loading:
```powershell
# Replace with your actual path if different
C:\Users\wenju\anaconda3\envs\sharp\python.exe -u bridge/sharp_bridge.py --mode file --max-points 100000 --overwrite
```

### TouchDesigner Setup
1. **Export Frames**: Use `td/scripts/frame_exporter.py` in a Script TOP.
2. **Render Splats**: Use a Gaussian Splat .tox and point it to `splats/current.ply`.
3. **Auto-Reload**: Use `td/scripts/splat_loader.py` to trigger reloads when the bridge updates.

## Project Structure
- `bridge/`: Core Python bridge and SHARP wrapper.
- `td/scripts/`: Integration scripts for TouchDesigner.
- `frames/`: Input buffer for image frames.
- `splats/`: Output folder for `.ply` Gaussian Splat files.

## License
[MIT](LICENSE) - Created by Wenjun (2026)

## Acknowledgements
- [Apple ML-SHARP](https://github.com/apple/ml-sharp) for the monocular view synthesis model.
- [StreamDiffusionTD](https://github.com/dotsimulate/StreamDiffusionTD) for the real-time image source.
